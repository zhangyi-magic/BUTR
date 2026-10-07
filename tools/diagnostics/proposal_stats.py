"""Optional proposal plots for decoders exposing proposal_stats_buffer."""

import torch


def compute_thresholds_by_percentile(areas_by_level, low_q=2.0, high_q=98.0):
    """
    areas_by_level: List of tensors, length >= 3
    For L levels, compute L-1 thresholds by:
      T_i = 0.5 * (p_high(L_i) + p_low(L_(i+1)))
    """

    def pct(tensor, q):
        if tensor.numel() == 0:
            return 0.0
        return float(torch.quantile(tensor, q / 100.0).item())

    L = len(areas_by_level)
    if L < 2:
        return []

    thresholds = []
    for i in range(L - 1):
        p_high = pct(areas_by_level[i], high_q)
        p_low = pct(areas_by_level[i + 1], low_q)
        thresholds.append(0.5 * (p_high + p_low))

    return thresholds


def save_proposal_stats_pdf(model, epoch, save_path, max_points=10240):
    """
    从 decoder.proposal_stats_buffer 收集 proposal 的 w, h, cls, ctr 和 level，
    绘制四类图并保存为 PDF：
    1. BBox宽高 vs feature level（箱线 + 散点 + 每个level points个数）
    2. BBox面积 vs feature level（箱线 + 散点 + 每个level统计信息）
    3. 类别 vs feature level（热力图 + 每个level points个数）
    4. 目标中心位置 vs feature level（散点图，level不同颜色 + 每个level points个数）
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import torch
    import numpy as np

    # 获取 decoder
    decoder = getattr(getattr(model, "module", model), "decoder", None)
    if decoder is None or not hasattr(decoder, "proposal_stats_buffer"):
        print(f"[Epoch {epoch}] No proposal_stats_buffer found in decoder.")
        return

    buffer = decoder.proposal_stats_buffer
    if not buffer:
        print(f"[Epoch {epoch}] proposal_stats_buffer is empty.")
        return

    # 合并 buffer
    all_w = torch.cat([d["w"] for d in buffer], dim=0)
    all_h = torch.cat([d["h"] for d in buffer], dim=0)
    all_level = torch.cat([d["level"] for d in buffer], dim=0)
    all_cls = torch.cat([d["cls"] for d in buffer], dim=0)
    all_ctr = torch.cat([d["ctr"] for d in buffer], dim=0)  # [N, 2]

    # 清空 buffer
    buffer.clear()

    # 采样 max_points
    total_points = all_w.shape[0]
    if total_points > max_points:
        idx = torch.linspace(0, total_points - 1, steps=max_points).long()
        all_w = all_w[idx]
        all_h = all_h[idx]
        all_level = all_level[idx]
        all_cls = all_cls[idx]
        all_ctr = all_ctr[idx]

    # 计算每个level的points个数
    levels = sorted(all_level.unique().tolist())
    level_counts = {}
    for lvl in levels:
        level_counts[lvl] = (all_level == lvl).sum().item()

    print(f"[Epoch {epoch}] Level counts: {level_counts}")

    # ============ 面积分位数阈值 ============
    areas = all_w * all_h
    areas_by_level = [areas[all_level == lvl] for lvl in levels]

    thresholds = compute_thresholds_by_percentile(areas_by_level, low_q=5.0, high_q=95.0)

    if len(thresholds) > 0:
        thr_str = ", ".join([f"T{i + 1}={v:.6f}" for i, v in enumerate(thresholds)])
        print(f"[Epoch {epoch}] Computed thresholds: {thr_str}")
    else:
        print(f"[Epoch {epoch}] Not enough levels to compute thresholds (need >= 2)")

    # ============ 宽高/面积 vs feature level ============
    jitter = 0.08
    offset = 0.18
    jittered_level_w = (
        all_level.float() - offset + (torch.rand_like(all_level.float()) - 0.5) * jitter
    )
    jittered_level_h = (
        all_level.float() + offset + (torch.rand_like(all_level.float()) - 0.5) * jitter
    )

    colors = {
        "width_scatter": "#4C72B0",
        "height_scatter": "#DD8452",
        "width_box": "#A6CEE3",
        "height_box": "#F5B7A9",
        "width_median": "#1F3552",
        "height_median": "#7F2F2F",
        "width_whisker": "#305080",
        "height_whisker": "#994444",
    }

    def plot_box_scatter(
        y_w, y_h, y_label, save_path_suffix, show_stats=False, show_level_counts=True
    ):
        plt.figure(figsize=(8, 6))
        levels = sorted(all_level.unique().tolist())
        width_data = [y_w[all_level == lvl].numpy() for lvl in levels]
        height_data = [y_h[all_level == lvl].numpy() for lvl in levels]

        plt.boxplot(
            width_data,
            positions=[lvl - offset for lvl in levels],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=colors["width_box"], edgecolor=colors["width_scatter"], alpha=0.6
            ),
            medianprops=dict(color=colors["width_median"], linewidth=2),
            whiskerprops=dict(color=colors["width_whisker"], linewidth=1.3),
            capprops=dict(color=colors["width_whisker"], linewidth=1.3),
            showfliers=False,
            zorder=3,
        )
        plt.boxplot(
            height_data,
            positions=[lvl + offset for lvl in levels],
            widths=0.25,
            patch_artist=True,
            boxprops=dict(
                facecolor=colors["height_box"], edgecolor=colors["height_scatter"], alpha=0.6
            ),
            medianprops=dict(color=colors["height_median"], linewidth=2),
            whiskerprops=dict(color=colors["height_whisker"], linewidth=1.3),
            capprops=dict(color=colors["height_whisker"], linewidth=1.3),
            showfliers=False,
            zorder=3,
        )

        plt.scatter(
            jittered_level_w,
            y_w,
            c=colors["width_scatter"],
            s=8,
            alpha=0.35,
            marker="o",
            label="Width",
            rasterized=True,
            zorder=1,
        )
        plt.scatter(
            jittered_level_h,
            y_h,
            c=colors["height_scatter"],
            s=8,
            alpha=0.35,
            marker="^",
            label="Height",
            rasterized=True,
            zorder=1,
        )

        plt.xticks(ticks=levels, labels=[str(lvl) for lvl in levels])
        plt.xlabel("Feature Level")
        plt.ylabel(y_label)
        if y_label != "BBox area (normalized)":
            plt.title(f"Proposal Width/Height vs Feature Level - Epoch {epoch}")
        else:
            plt.title(f"Proposal Area vs Feature Level - Epoch {epoch}")

        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.5)

        # 添加统计信息
        stats_text = []
        if show_stats:
            for lvl in levels:
                data = y_w[all_level == lvl].numpy()  # y_w == y_h == area
                if len(data) > 0:
                    mean_val = np.mean(data)
                    std_val = np.std(data)
                    median_val = np.median(data)
                    min_val = np.min(data)
                    max_val = np.max(data)
                    count = len(data)
                    stats_text.append(
                        f"Level {lvl}: N={count}, μ={mean_val:.4f}, σ={std_val:.4f}, "
                        f"med={median_val:.4f}, min={min_val:.4f}, max={max_val:.4f}"
                    )
        elif show_level_counts:
            # 只显示每个level的points个数
            for lvl in levels:
                count = level_counts[lvl]
                stats_text.append(f"Level {lvl}: N={count}")

        if stats_text:
            text_str = "\n".join(stats_text)
            plt.gcf().text(
                0.98,
                0.02,
                text_str,
                ha="right",
                va="bottom",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
            )

        save_path_full = save_path.parent / f"{save_path.stem}{save_path_suffix}.pdf"
        plt.savefig(save_path_full, bbox_inches="tight")
        plt.close()
        print(f"[Epoch {epoch}] Proposal stats saved to {save_path_full}")

    # 宽高图 + 显示每个level的points个数
    plot_box_scatter(all_w, all_h, "BBox size (normalized)", "", show_level_counts=True)
    # 面积图 + 显示完整统计信息
    plot_box_scatter(
        all_w * all_h, all_w * all_h, "BBox area (normalized)", "_area", show_stats=True
    )

    # ============ 类别 vs level 热力图 ============
    plt.figure(figsize=(10, 6))  # 增加宽度以容纳level counts信息
    num_levels = int(all_level.max().item()) + 1
    num_classes = int(all_cls.max().item()) + 1
    heatmap = torch.zeros((num_classes, num_levels), dtype=torch.int32)
    for cls_val, lvl_val in zip(all_cls, all_level):
        heatmap[cls_val.long(), lvl_val.long()] += 1

    plt.imshow(heatmap.numpy(), cmap="Blues", origin="lower", aspect="auto")
    plt.colorbar(label="Num proposals")
    plt.xlabel("Feature Level")
    plt.ylabel("Class")
    plt.title(f"Class vs Feature Level - Epoch {epoch}")
    plt.xticks(ticks=range(num_levels), labels=[str(i) for i in range(num_levels)])
    plt.yticks(ticks=range(num_classes), labels=[str(i) for i in range(num_classes)])

    # 在热力图上添加每个level的points个数
    for lvl in range(num_levels):
        count = level_counts.get(lvl, 0)
        plt.text(
            lvl,
            num_classes - 0.5,
            f"N={count}",
            ha="center",
            va="bottom",
            fontsize=8,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"),
        )

    save_path_full = save_path.parent / f"{save_path.stem}_cls_level.pdf"
    plt.savefig(save_path_full, bbox_inches="tight")
    plt.close()
    print(f"[Epoch {epoch}] Class vs Feature Level heatmap saved to {save_path_full}")

    # ============ 目标中心位置 vs level ============
    plt.figure(figsize=(8, 6))  # 增加尺寸以容纳图例和统计信息
    num_levels = int(all_level.max().item()) + 1
    cmap = cm.get_cmap("tab10", num_levels)

    all_ctr_flat = all_ctr.reshape(-1, 2)
    all_level_flat = all_level.reshape(-1)

    for lvl in range(num_levels):
        mask = all_level_flat == lvl
        count = level_counts.get(lvl, 0)
        plt.scatter(
            all_ctr_flat[mask, 0].numpy(),
            all_ctr_flat[mask, 1].numpy(),
            color=cmap(lvl),
            s=4,
            alpha=0.25,
            label=f"Level {lvl} (N={count})",
        )

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("cx (normalized)")
    plt.ylabel("cy (normalized)")
    plt.title(f"Proposal center vs Feature Level - Epoch {epoch}")

    # 调整图例位置和大小以避免重叠
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0, fontsize=8)

    plt.grid(True, linestyle="--", alpha=0.5)

    # 在图上添加总points信息
    total_count = sum(level_counts.values())
    plt.figtext(
        0.02,
        0.02,
        f"Total points: {total_count}",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.8),
    )

    save_path_full = save_path.parent / f"{save_path.stem}_center_level.pdf"
    plt.savefig(save_path_full, bbox_inches="tight")
    plt.close()
    print(f"[Epoch {epoch}] Center vs Feature Level scatter saved to {save_path_full}")
