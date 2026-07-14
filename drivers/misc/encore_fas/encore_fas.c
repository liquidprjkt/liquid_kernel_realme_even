#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/uprobes.h>
#include <linux/hrtimer.h>
#include <linux/ktime.h>
#include <linux/idr.h>
#include <linux/slab.h>
#include <linux/spinlock.h>
#include <linux/mutex.h>
#include <linux/wait.h>
#include <linux/poll.h>
#include <linux/kfifo.h>
#include <linux/kref.h>
#include <linux/pid.h>
#include <linux/sched.h>
#include <linux/workqueue.h>
#include <linux/namei.h>
#include <linux/version.h>
#include <linux/cred.h>
#include <linux/string.h>
#include <trace/events/sched.h>

#include "encore_fas_uapi.h"

#ifndef __aarch64__
#error "This kernel module is strictly restricted to ARM64 (AArch64) systems!"
#endif

#define FAS_EVENT_FIFO_SIZE   1024
#define FAS_DEFAULT_FPS       60
#define FAS_SMALL_MULT        2
#define FAS_BIG_MULT          5
#define FAS_BIG_JANK_FLOOR_MS 200
#define FAS_EMA_DIVISOR       5
#define FAS_RELOCK_STREAK     3
#define FAS_RELOCK_TOLERANCE_PCT 20

// ==========================================
// ARM64 Hardware Counter
// ==========================================

static uint64_t g_counter_freq = 0;

static inline uint64_t read_virtual_counter(void) {
    uint64_t val;
    asm volatile("mrs %0, cntvct_el0" : "=r"(val));
    return val;
}

static inline uint64_t read_counter_freq(void) {
    uint64_t freq;
    asm volatile("mrs %0, cntfrq_el0" : "=r"(freq));
    return freq;
}

static inline uint64_t counter_to_ms(uint64_t counter) {
    if (g_counter_freq == 0) return 0;
    return (counter * 1000) / g_counter_freq;
}

static inline uint64_t fas_now_ms(void) {
    return counter_to_ms(read_virtual_counter());
}

// ==========================================
// Global state
// ==========================================

struct fas_context {
    struct kref kref;
    atomic_t active;
    int id;
    struct pid *pid_struct;
    pid_t pid;

    struct path libgui_path;
    bool uprobe_registered;
    struct uprobe_consumer consumer;

    struct hrtimer watchdog_timer;
    uint64_t baseline_ms;
    uint64_t last_frame_ms;
    int watchdog_stage;
    uint32_t relock_count;
    uint64_t relock_deltas[FAS_RELOCK_STREAK];

    struct work_struct teardown_work;
    struct list_head node;
};

static DEFINE_IDR(fas_ctx_idr);
static DEFINE_MUTEX(fas_ctx_lock);
static LIST_HEAD(fas_ctx_list);

static DEFINE_MUTEX(fas_libgui_lock);
static uint64_t g_libgui_offset;
static char g_libgui_path[FAS_MAX_PATH_LEN];

static DEFINE_SPINLOCK(fas_fifo_lock);
static DECLARE_KFIFO(fas_event_fifo, struct fas_jank_event, FAS_EVENT_FIFO_SIZE);
static DECLARE_WAIT_QUEUE_HEAD(fas_poll_wait);

static struct workqueue_struct *fas_teardown_wq;

// ==========================================
// Version compatibility helpers
// ==========================================

/**
 * @brief Unregisters a uprobe consumer across the kernel API split introduced
 *        in 6.9 where teardown became a two step nosync/sync operation.
 * @param inode Inode the uprobe was registered against.
 * @param offset File offset of the probed instruction.
 * @param uc Consumer to unregister.
 */
static inline void fas_uprobe_unregister(struct inode *inode, loff_t offset,
                                          struct uprobe_consumer *uc) {
#if LINUX_VERSION_CODE >= KERNEL_VERSION(6, 9, 0)
    uprobe_unregister_nosync(inode, offset, uc);
    uprobe_unregister_sync();
#else
    uprobe_unregister(inode, offset, uc);
#endif
}

static inline bool fas_caller_is_root(void) {
    return uid_eq(current_euid(), GLOBAL_ROOT_UID);
}

// ==========================================
// Jank detection
// ==========================================

static inline uint64_t fas_small_threshold(struct fas_context *ctx) {
    return ctx->baseline_ms * FAS_SMALL_MULT;
}

static inline uint64_t fas_big_threshold(struct fas_context *ctx) {
    return max_t(uint64_t, ctx->baseline_ms * FAS_BIG_MULT, FAS_BIG_JANK_FLOOR_MS);
}

static void fas_ema_update(struct fas_context *ctx, uint64_t delta) {
    int64_t diff = (int64_t)delta - (int64_t)ctx->baseline_ms;

    ctx->baseline_ms = (uint64_t)((int64_t)ctx->baseline_ms + diff / FAS_EMA_DIVISOR);
    if (ctx->baseline_ms == 0)
        ctx->baseline_ms = 1;
}

static bool fas_deltas_close(uint64_t a, uint64_t b, uint32_t pct) {
    uint64_t diff = a > b ? a - b : b - a;
    uint64_t allowed = (a * pct) / 100;

    return diff <= allowed;
}

/**
 * @brief Tracks a short run of similarly-sized jank-range deltas. A single
 *        dropped frame does not move the baseline, but a sustained cadence
 *        change (mode switch, thermal cap, deliberately slow rendering)
 *        snaps the baseline onto the new cluster within a few frames instead
 *        of waiting for the EMA to slowly drift there.
 */
static void fas_relock_update(struct fas_context *ctx, uint64_t delta) {
    if (ctx->relock_count > 0 &&
        !fas_deltas_close(ctx->relock_deltas[ctx->relock_count - 1], delta, FAS_RELOCK_TOLERANCE_PCT))
        ctx->relock_count = 0;

    ctx->relock_deltas[ctx->relock_count % FAS_RELOCK_STREAK] = delta;
    ctx->relock_count++;

    if (ctx->relock_count >= FAS_RELOCK_STREAK) {
        uint64_t sum = 0;
        int i;

        for (i = 0; i < FAS_RELOCK_STREAK; i++)
            sum += ctx->relock_deltas[i];

        ctx->baseline_ms = sum / FAS_RELOCK_STREAK;
        ctx->relock_count = 0;
    }
}

/**
 * @brief Arms the watchdog for the next frame gap, timed off the current
 *        adaptive baseline rather than a fixed guess.
 */
static void fas_arm_watchdog(struct fas_context *ctx) {
    ctx->watchdog_stage = 0;
    hrtimer_start(&ctx->watchdog_timer, ms_to_ktime(fas_small_threshold(ctx)), HRTIMER_MODE_REL);
}

/**
 * @brief Queues a jank event onto the shared fifo and wakes any readers
 *        blocked on poll()/read().
 * @param ctx_id Listener handle the event belongs to.
 * @param type FAS_EVENT_* classification.
 * @param frametime_ms Observed frametime that triggered the event.
 */
static void fas_push_event(int ctx_id, enum fas_event_type type, uint64_t frametime_ms) {
    struct fas_jank_event ev = {
        .ctx_id = ctx_id,
        .type = type,
        .timestamp_ms = fas_now_ms(),
        .frametime_ms = frametime_ms,
    };
    unsigned long flags;

    spin_lock_irqsave(&fas_fifo_lock, flags);
    kfifo_put(&fas_event_fifo, ev);
    spin_unlock_irqrestore(&fas_fifo_lock, flags);

    wake_up_interruptible(&fas_poll_wait);
}

/**
 * @brief Uprobe hit handler for Surface::queueBuffer. Classifies the gap
 *        since the previous frame against the adaptive baseline, feeds the
 *        baseline (EMA on normal frames, relock streak on sustained jank
 *        ranges), and re-arms the watchdog for the next gap. Runs in
 *        breakpoint trap context of the traced task; every call here is
 *        non-sleeping (hrtimer_cancel/start, spinlock, plain arithmetic).
 */
static int fas_uprobe_handler(struct uprobe_consumer *self, struct pt_regs *regs) {
    struct fas_context *ctx = container_of(self, struct fas_context, consumer);
    uint64_t now, delta;

    if (!atomic_read(&ctx->active))
        return 0;

    now = fas_now_ms();
    delta = now - ctx->last_frame_ms;
    ctx->last_frame_ms = now;

    hrtimer_cancel(&ctx->watchdog_timer);

    if (delta <= fas_small_threshold(ctx)) {
        fas_ema_update(ctx, delta);
        ctx->relock_count = 0;
    } else if (delta <= fas_big_threshold(ctx)) {
        fas_push_event(ctx->id, FAS_EVENT_SMALL_JANK, delta);
        fas_relock_update(ctx, delta);
    } else {
        fas_push_event(ctx->id, FAS_EVENT_BIG_JANK, delta);
        fas_relock_update(ctx, delta);
    }

    fas_arm_watchdog(ctx);
    return 0;
}

/**
 * @brief Restricts this consumer's uprobe hits to the traced task only.
 */
static bool fas_uprobe_filter(struct uprobe_consumer *self, enum uprobe_filter_ctx ctx,
                               struct mm_struct *mm) {
    struct fas_context *fctx = container_of(self, struct fas_context, consumer);
    struct task_struct *task = pid_task(fctx->pid_struct, PIDTYPE_PID);

    return task && task->mm == mm;
}

/**
 * @brief Proactive boost signal, timed off the adaptive baseline. Fires at
 *        most twice per frame gap (soft at small_threshold, hard at
 *        big_threshold) and is cancelled the instant the real frame lands,
 *        so a consistently slow app only ever gets one soft/hard pair per
 *        gap instead of an escalating storm.
 */
static enum hrtimer_restart fas_watchdog_fn(struct hrtimer *timer) {
    struct fas_context *ctx = container_of(timer, struct fas_context, watchdog_timer);
    uint64_t elapsed;

    if (!atomic_read(&ctx->active))
        return HRTIMER_NORESTART;

    elapsed = fas_now_ms() - ctx->last_frame_ms;

    if (ctx->watchdog_stage == 0) {
        fas_push_event(ctx->id, FAS_EVENT_BOOST_SOFT, elapsed);
        ctx->watchdog_stage = 1;
        hrtimer_forward_now(timer,
            ms_to_ktime(fas_big_threshold(ctx) - fas_small_threshold(ctx)));
        return HRTIMER_RESTART;
    }

    fas_push_event(ctx->id, FAS_EVENT_BOOST_HARD, elapsed);
    ctx->watchdog_stage = 2;
    return HRTIMER_NORESTART;
}

// ==========================================
// Context lifecycle
// ==========================================

static void fas_context_release(struct kref *kref) {
    struct fas_context *ctx = container_of(kref, struct fas_context, kref);

    put_pid(ctx->pid_struct);
    kfree(ctx);
}

static void fas_context_put(struct fas_context *ctx) {
    kref_put(&ctx->kref, fas_context_release);
}

/**
 * @brief Looks up a listener by handle and takes a reference so it cannot be
 *        freed by a concurrent teardown (e.g. the traced process exiting)
 *        while the caller is still using it. Caller must fas_context_put().
 */
static struct fas_context *fas_ctx_lookup_get(int ctx_id) {
    struct fas_context *ctx;

    mutex_lock(&fas_ctx_lock);
    ctx = idr_find(&fas_ctx_idr, ctx_id);
    if (ctx && !kref_get_unless_zero(&ctx->kref))
        ctx = NULL;
    mutex_unlock(&fas_ctx_lock);

    return ctx;
}

/**
 * @brief Tears down uprobe, timer and path reference for a listener. Safe to
 *        call from either the ioctl removal path or the async exit-tracking
 *        workqueue; the atomic active flag makes it idempotent.
 */
static void fas_context_teardown(struct fas_context *ctx) {
    if (atomic_cmpxchg(&ctx->active, 1, 0) != 1)
        return;

    if (ctx->uprobe_registered) {
        fas_uprobe_unregister(d_inode(ctx->libgui_path.dentry), g_libgui_offset, &ctx->consumer);
        ctx->uprobe_registered = false;
    }

    /*
     * uprobe_unregister() above blocks until any in-flight handler
     * invocation has fully returned, so no handler can still be racing to
     * re-arm the watchdog by the time we cancel it here. Cancelling first
     * would leave a window where an in-flight handler re-arms the timer
     * after our cancel but before unregister finishes waiting for it.
     */
    hrtimer_cancel(&ctx->watchdog_timer);

    path_put(&ctx->libgui_path);

    mutex_lock(&fas_ctx_lock);
    idr_remove(&fas_ctx_idr, ctx->id);
    list_del(&ctx->node);
    mutex_unlock(&fas_ctx_lock);

    fas_context_put(ctx);
}

static void fas_teardown_work_fn(struct work_struct *work) {
    struct fas_context *ctx = container_of(work, struct fas_context, teardown_work);

    fas_context_teardown(ctx);
    fas_context_put(ctx);
}

/**
 * @brief sched_process_exit tracepoint callback. Auto-removes any listener
 *        whose traced task exited without an explicit remove_jank_listener
 *        call, so listeners can never outlive their target process.
 */
static void fas_process_exit_probe(void *data, struct task_struct *task) {
    struct fas_context *ctx;

    mutex_lock(&fas_ctx_lock);
    list_for_each_entry(ctx, &fas_ctx_list, node) {
        if (pid_task(ctx->pid_struct, PIDTYPE_PID) == task && atomic_read(&ctx->active)) {
            kref_get(&ctx->kref);
            queue_work(fas_teardown_wq, &ctx->teardown_work);
            break;
        }
    }
    mutex_unlock(&fas_ctx_lock);
}

/**
 * @brief Seeds (or re-seeds) the adaptive baseline from a hinted fps. This is
 *        only a cold-start value now, real classification thresholds are
 *        derived from baseline_ms which the uprobe handler continuously
 *        corrects from observed frame intervals.
 */
static void fas_apply_fps_hint(struct fas_context *ctx, uint32_t fps) {
    uint64_t ms = fps ? (1000 / fps) : (1000 / FAS_DEFAULT_FPS);

    ctx->baseline_ms = ms ? ms : 1;
}

// ==========================================
// ioctl interface
// ==========================================

static long fas_ioctl_update_offset(void __user *arg) {
    struct fas_libgui_offset req;

    if (copy_from_user(&req, arg, sizeof(req)))
        return -EFAULT;

    req.path[FAS_MAX_PATH_LEN - 1] = '\0';

    mutex_lock(&fas_libgui_lock);
    g_libgui_offset = req.offset;
    strscpy(g_libgui_path, req.path, FAS_MAX_PATH_LEN);
    mutex_unlock(&fas_libgui_lock);

    return 0;
}

static long fas_ioctl_get_offset(void __user *arg) {
    struct fas_libgui_offset resp;

    mutex_lock(&fas_libgui_lock);
    resp.offset = g_libgui_offset;
    strscpy(resp.path, g_libgui_path, FAS_MAX_PATH_LEN);
    mutex_unlock(&fas_libgui_lock);

    if (copy_to_user(arg, &resp, sizeof(resp)))
        return -EFAULT;

    return 0;
}

/**
 * @brief Allocates a listener context, attaches the queueBuffer uprobe
 *        filtered to the requested pid, seeds the adaptive baseline, and
 *        arms the boost watchdog.
 */
static long fas_ioctl_register_listener(void __user *arg) {
    struct fas_register_args req;
    struct fas_context *ctx;
    int ret;

    if (copy_from_user(&req, arg, sizeof(req)))
        return -EFAULT;

    mutex_lock(&fas_libgui_lock);
    if (g_libgui_path[0] == '\0') {
        mutex_unlock(&fas_libgui_lock);
        return -ENOENT;
    }
    mutex_unlock(&fas_libgui_lock);

    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return -ENOMEM;

    ctx->pid_struct = find_get_pid(req.pid);
    if (!ctx->pid_struct) {
        ret = -ESRCH;
        goto err_free;
    }
    ctx->pid = req.pid;

    ret = kern_path(g_libgui_path, LOOKUP_FOLLOW, &ctx->libgui_path);
    if (ret)
        goto err_put_pid;

    kref_init(&ctx->kref);
    atomic_set(&ctx->active, 1);
    ctx->last_frame_ms = fas_now_ms();
    ctx->watchdog_stage = 0;
    ctx->relock_count = 0;
    INIT_WORK(&ctx->teardown_work, fas_teardown_work_fn);
    fas_apply_fps_hint(ctx, FAS_DEFAULT_FPS);

    ctx->consumer.handler = fas_uprobe_handler;
    ctx->consumer.filter = fas_uprobe_filter;

    mutex_lock(&fas_ctx_lock);
    ret = idr_alloc(&fas_ctx_idr, ctx, 1, 0, GFP_KERNEL);
    mutex_unlock(&fas_ctx_lock);
    if (ret < 0)
        goto err_put_path;
    ctx->id = ret;

    ret = uprobe_register(d_inode(ctx->libgui_path.dentry), g_libgui_offset, &ctx->consumer);
    if (ret)
        goto err_remove_idr;
    ctx->uprobe_registered = true;

    hrtimer_init(&ctx->watchdog_timer, CLOCK_MONOTONIC, HRTIMER_MODE_REL);
    ctx->watchdog_timer.function = fas_watchdog_fn;

    mutex_lock(&fas_ctx_lock);
    list_add_tail(&ctx->node, &fas_ctx_list);
    mutex_unlock(&fas_ctx_lock);

    fas_arm_watchdog(ctx);

    req.ctx_id = ctx->id;
    if (copy_to_user(arg, &req, sizeof(req))) {
        fas_context_teardown(ctx);
        return -EFAULT;
    }

    return 0;

err_remove_idr:
    mutex_lock(&fas_ctx_lock);
    idr_remove(&fas_ctx_idr, ctx->id);
    mutex_unlock(&fas_ctx_lock);
err_put_path:
    path_put(&ctx->libgui_path);
err_put_pid:
    put_pid(ctx->pid_struct);
err_free:
    kfree(ctx);
    return ret;
}

static long fas_ioctl_remove_listener(void __user *arg) {
    int32_t ctx_id;
    struct fas_context *ctx;

    if (copy_from_user(&ctx_id, arg, sizeof(ctx_id)))
        return -EFAULT;

    ctx = fas_ctx_lookup_get(ctx_id);
    if (!ctx)
        return -ENOENT;

    fas_context_teardown(ctx);
    fas_context_put(ctx);
    return 0;
}

static long fas_ioctl_hint_frametime(void __user *arg) {
    struct fas_frametime_hint req;
    struct fas_context *ctx;

    if (copy_from_user(&req, arg, sizeof(req)))
        return -EFAULT;

    ctx = fas_ctx_lookup_get(req.ctx_id);
    if (!ctx)
        return -ENOENT;

    if (!atomic_read(&ctx->active)) {
        fas_context_put(ctx);
        return -ENOENT;
    }

    fas_apply_fps_hint(ctx, req.fps);
    fas_context_put(ctx);
    return 0;
}

static long fas_ioctl(struct file *file, unsigned int cmd, unsigned long arg) {
    void __user *uarg = (void __user *)arg;

    if (!fas_caller_is_root())
        return -EPERM;

    switch (cmd) {
    case FAS_IOC_UPDATE_LIBGUI_OFFSET:
        return fas_ioctl_update_offset(uarg);
    case FAS_IOC_GET_LIBGUI_OFFSET:
        return fas_ioctl_get_offset(uarg);
    case FAS_IOC_REGISTER_LISTENER:
        return fas_ioctl_register_listener(uarg);
    case FAS_IOC_REMOVE_LISTENER:
        return fas_ioctl_remove_listener(uarg);
    case FAS_IOC_HINT_FRAMETIME:
        return fas_ioctl_hint_frametime(uarg);
    default:
        return -ENOTTY;
    }
}

static int fas_open(struct inode *inode, struct file *file) {
    if (!fas_caller_is_root())
        return -EPERM;

    return nonseekable_open(inode, file);
}

static ssize_t fas_read(struct file *file, char __user *buf, size_t count, loff_t *ppos) {
    struct fas_jank_event ev;
    unsigned long flags;
    unsigned int copied;
    int ret;

    if (count < sizeof(ev))
        return -EINVAL;

    while (true) {
        spin_lock_irqsave(&fas_fifo_lock, flags);
        ret = kfifo_get(&fas_event_fifo, &ev);
        spin_unlock_irqrestore(&fas_fifo_lock, flags);

        if (ret)
            break;

        if (file->f_flags & O_NONBLOCK)
            return -EAGAIN;

        if (wait_event_interruptible(fas_poll_wait, !kfifo_is_empty(&fas_event_fifo)))
            return -ERESTARTSYS;
    }

    copied = sizeof(ev);
    if (copy_to_user(buf, &ev, copied))
        return -EFAULT;

    return copied;
}

static unsigned int fas_poll(struct file *file, poll_table *wait) {
    poll_wait(file, &fas_poll_wait, wait);

    if (!kfifo_is_empty(&fas_event_fifo))
        return POLLIN | POLLRDNORM;

    return 0;
}

static const struct file_operations fas_fops = {
    .owner = THIS_MODULE,
    .open = fas_open,
    .unlocked_ioctl = fas_ioctl,
    .read = fas_read,
    .poll = fas_poll,
};

static struct miscdevice fas_miscdev = {
    .minor = MISC_DYNAMIC_MINOR,
    .name = "encore_fas",
    .fops = &fas_fops,
    .mode = 0600,
};

// ==========================================
// Module Initialization and Exit
// ==========================================

static int __init encore_fas_init(void) {
    int ret;

    g_counter_freq = read_counter_freq();
    INIT_KFIFO(fas_event_fifo);

    fas_teardown_wq = alloc_workqueue("encore_fas_teardown", WQ_UNBOUND, 0);
    if (!fas_teardown_wq)
        return -ENOMEM;

    ret = misc_register(&fas_miscdev);
    if (ret) {
        destroy_workqueue(fas_teardown_wq);
        return ret;
    }

    ret = register_trace_sched_process_exit(fas_process_exit_probe, NULL);
    if (ret) {
        misc_deregister(&fas_miscdev);
        destroy_workqueue(fas_teardown_wq);
        return ret;
    }

    return 0;
}

static void __exit encore_fas_exit(void) {
    struct fas_context *ctx, *tmp;

    unregister_trace_sched_process_exit(fas_process_exit_probe, NULL);
    tracepoint_synchronize_unregister();

    mutex_lock(&fas_ctx_lock);
    list_for_each_entry_safe(ctx, tmp, &fas_ctx_list, node) {
        mutex_unlock(&fas_ctx_lock);
        fas_context_teardown(ctx);
        mutex_lock(&fas_ctx_lock);
    }
    mutex_unlock(&fas_ctx_lock);

    destroy_workqueue(fas_teardown_wq);
    misc_deregister(&fas_miscdev);
}

module_init(encore_fas_init);
module_exit(encore_fas_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Rem01Gaming");
MODULE_DESCRIPTION("Encore Tweaks FAS Kernel Module");
MODULE_VERSION("0.1");
