"""餐厅后厨排产与可视化示例。

功能：
1. 使用 OR-Tools 做双灶台排产。
2. 在终端打印排产表。
3. 生成适合文章展示的甘特图 PNG。

运行示例：
    python OrCooking.py
    python OrCooking.py --save-path output/cooking_gantt.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from ortools.sat.python import cp_model
except ImportError:
    cp_model = None

try:
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.patches import Patch
except ImportError:
    plt = None
    font_manager = None
    Patch = None


MENU = {
    "A": {"duration": 20, "flavor": "蒸", "profit": 50},
    "B": {"duration": 15, "flavor": "蒸", "profit": 40},
    "C": {"duration": 8, "flavor": "炒", "profit": 15},
    "D": {"duration": 5, "flavor": "炒", "profit": 10},
    "E": {"duration": 6, "flavor": "炒", "profit": 20},
    "F": {"duration": 7, "flavor": "炒", "profit": 12},
    "G": {"duration": 4, "flavor": "煎", "profit": 8},
    "H": {"duration": 10, "flavor": "蒸", "profit": 30},
    "I": {"duration": 5, "flavor": "炒", "profit": 15},
    "J": {"duration": 6, "flavor": "炒", "profit": 12},
}

TABLES = {
    1: {"orders": ["A", "B", "C", "D"], "dist": 10, "priority": 1.0},
    2: {"orders": ["A", "E", "F", "G"], "dist": 30, "priority": 1.0},
    3: {"orders": ["B", "H", "I", "J"], "dist": 60, "priority": 1.5},
}

FLAVOR_COLORS = {
    "蒸": "#5B8FF9",
    "炒": "#F6BD16",
    "煎": "#5AD8A6",
}


def parse_args():
    parser = argparse.ArgumentParser(description="餐厅后厨排产与甘特图可视化")
    parser.add_argument(
        "--save-path",
        default="OR/cooking_gantt.png",
        help="甘特图输出路径，默认保存到 OR/cooking_gantt.png",
    )
    return parser.parse_args()


def build_tasks():
    tasks = []
    for table_id, table_info in TABLES.items():
        for dish in table_info["orders"]:
            dish_info = MENU[dish]
            tasks.append(
                {
                    "table": table_id,
                    "dish": dish,
                    "duration": dish_info["duration"],
                    "flavor": dish_info["flavor"],
                    "profit": dish_info["profit"],
                    "distance": table_info["dist"],
                    "priority": table_info["priority"],
                    "weight": table_info["priority"] * (1 + table_info["dist"] / 100),
                }
            )
    return tasks


def format_results(rows):
    headers = ["桌号", "菜品", "灶台", "开始时间", "结束时间", "口味", "距离", "权重"]
    widths = []
    for idx, header in enumerate(headers):
        width = len(str(header))
        for row in rows:
            width = max(width, len(str(row[idx])))
        widths.append(width)

    header_line = "  ".join(str(text).ljust(widths[i]) for i, text in enumerate(headers))
    split_line = "  ".join("-" * width for width in widths)
    body = [
        "  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row))
        for row in rows
    ]
    return "\n".join([header_line, split_line, *body])


def solve_schedule(tasks, num_cookers=2, setup_time=5):
    model = cp_model.CpModel()
    num_tasks = len(tasks)
    horizon = sum(task["duration"] for task in tasks) + num_tasks * setup_time

    starts = [model.NewIntVar(0, horizon, f"start_{i}") for i in range(num_tasks)]
    ends = [model.NewIntVar(0, horizon, f"end_{i}") for i in range(num_tasks)]
    cookers = [model.NewIntVar(0, num_cookers - 1, f"cooker_{i}") for i in range(num_tasks)]

    for i, task in enumerate(tasks):
        model.Add(ends[i] == starts[i] + task["duration"])

    for cooker in range(num_cookers):
        cooker_intervals = []
        for i, task in enumerate(tasks):
            assigned = model.NewBoolVar(f"task_{i}_on_cooker_{cooker}")
            model.Add(cookers[i] == cooker).OnlyEnforceIf(assigned)
            model.Add(cookers[i] != cooker).OnlyEnforceIf(assigned.Not())
            cooker_intervals.append(
                model.NewOptionalIntervalVar(
                    starts[i],
                    task["duration"],
                    ends[i],
                    assigned,
                    f"interval_{i}_{cooker}",
                )
            )
        model.AddNoOverlap(cooker_intervals)

    for i in range(num_tasks):
        for j in range(i + 1, num_tasks):
            same_cooker = model.NewBoolVar(f"same_cooker_{i}_{j}")
            model.Add(cookers[i] == cookers[j]).OnlyEnforceIf(same_cooker)
            model.Add(cookers[i] != cookers[j]).OnlyEnforceIf(same_cooker.Not())

            if tasks[i]["dish"] == tasks[j]["dish"]:
                model.Add(starts[i] == starts[j]).OnlyEnforceIf(same_cooker)
                continue

            i_before_j = model.NewBoolVar(f"i_before_j_{i}_{j}")
            gap = setup_time if tasks[i]["flavor"] != tasks[j]["flavor"] else 0
            model.Add(starts[j] >= ends[i] + gap).OnlyEnforceIf([same_cooker, i_before_j])
            model.Add(starts[i] >= ends[j] + gap).OnlyEnforceIf([same_cooker, i_before_j.Not()])

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, ends)
    weighted_finish = sum(ends[i] * int(tasks[i]["weight"] * 10) for i in range(num_tasks))
    model.Minimize(makespan * 50 + weighted_finish)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    results = []
    for i, task in enumerate(tasks):
        results.append(
            {
                **task,
                "task_id": i,
                "cooker": solver.Value(cookers[i]) + 1,
                "start": solver.Value(starts[i]),
                "end": solver.Value(ends[i]),
            }
        )

    results.sort(key=lambda item: (item["cooker"], item["start"], item["table"], item["dish"]))
    return {
        "results": results,
        "makespan": solver.Value(makespan),
        "setup_time": setup_time,
    }


def choose_chinese_font():
    if font_manager is None:
        return None

    candidates = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "WenQuanYi Zen Hei",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None


def find_batch_pairs(results):
    pairs = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            left = results[i]
            right = results[j]
            if (
                left["dish"] == right["dish"]
                and left["cooker"] == right["cooker"]
                and left["start"] == right["start"]
            ):
                pairs.append((left, right))
    return pairs


def find_setup_gap_pair(results, setup_time):
    by_cooker = {}
    for item in results:
        by_cooker.setdefault(item["cooker"], []).append(item)

    for cooker_results in by_cooker.values():
        cooker_results.sort(key=lambda item: item["start"])
        for left, right in zip(cooker_results, cooker_results[1:]):
            gap = right["start"] - left["end"]
            if gap >= setup_time and left["flavor"] != right["flavor"]:
                return left, right, gap
    return None


def create_gantt_chart(results, makespan, setup_time, save_path):
    if plt is None or Patch is None:
        print("未检测到 matplotlib，无法生成图形。请先安装：pip install matplotlib")
        return

    font_name = choose_chinese_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor("#F8F6F1")
    ax.set_facecolor("#FFFDF8")

    cooker_y = {1: 30, 2: 10}
    bar_height = 6

    for item in results:
        y = cooker_y[item["cooker"]]
        color = FLAVOR_COLORS[item["flavor"]]
        alpha = 1.0 if item["table"] == 3 else 0.88
        edge = "#B03A2E" if item["table"] == 3 else "#2F4858"

        ax.broken_barh(
            [(item["start"], item["duration"])],
            (y, bar_height),
            facecolors=color,
            edgecolors=edge,
            linewidth=2 if item["table"] == 3 else 1.2,
            alpha=alpha,
            zorder=3,
        )
        ax.text(
            item["start"] + item["duration"] / 2,
            y + bar_height / 2,
            f"桌{item['table']} {item['dish']}",
            ha="center",
            va="center",
            fontsize=10,
            color="#1F1F1F",
            zorder=4,
        )

    batch_pairs = find_batch_pairs(results)
    if batch_pairs:
        left, right = batch_pairs[0]
        y = cooker_y[left["cooker"]] + bar_height + 2
        start = left["start"]
        width = max(left["duration"], right["duration"])
        ax.annotate(
            "一锅出：桌1与桌2的同款菜被同时开做",
            xy=(start + width / 2, y),
            xytext=(start + 18, y + 10),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#D1495B"},
            fontsize=11,
            color="#8C1C13",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#FFE5D4", "ec": "#D1495B"},
            zorder=5,
        )

    setup_pair = find_setup_gap_pair(results, setup_time)
    if setup_pair:
        left, right, gap = setup_pair
        y = cooker_y[left["cooker"]] - 2
        ax.annotate(
            "",
            xy=(left["end"], y),
            xytext=(right["start"], y),
            arrowprops={"arrowstyle": "<->", "lw": 2, "color": "#7A5195"},
            zorder=5,
        )
        ax.text(
            (left["end"] + right["start"]) / 2,
            y - 2,
            f"洗锅/切换 {gap} 分钟",
            ha="center",
            va="top",
            fontsize=10,
            color="#5A189A",
            bbox={"boxstyle": "round,pad=0.25", "fc": "#F3E8FF", "ec": "#7A5195"},
            zorder=5,
        )

    vip_items = [item for item in results if item["table"] == 3]
    if vip_items:
        vip_item = min(vip_items, key=lambda item: item["start"])
        y = cooker_y[vip_item["cooker"]] + bar_height / 2
        ax.annotate(
            "远桌 VIP 权重更高\n算法会尽量让桌3更早入锅",
            xy=(vip_item["start"], y),
            xytext=(max(vip_item["start"] - 18, 2), y + 12),
            arrowprops={"arrowstyle": "->", "lw": 1.8, "color": "#C1121F"},
            fontsize=11,
            color="#780000",
            bbox={"boxstyle": "round,pad=0.35", "fc": "#FFF1E6", "ec": "#C1121F"},
            zorder=5,
        )

    ax.set_xlim(0, makespan + 10)
    ax.set_ylim(5, 42)
    ax.set_xticks(range(0, makespan + 11, 5))
    ax.set_yticks([13, 33])
    ax.set_yticklabels(["灶台 2", "灶台 1"])
    ax.grid(axis="x", linestyle="--", alpha=0.25, zorder=1)

    ax.set_title("餐厅后厨排产甘特图：一锅出、洗锅切换与远桌优先", fontsize=16, pad=20)
    ax.set_xlabel("时间（分钟）", fontsize=12)
    ax.set_ylabel("后厨资源", fontsize=12)

    summary = (
        "图示解读：同款菜同时开做体现“一锅出”；不同口味之间预留 5 分钟切换；"
        "桌3因距离远且优先级高，被赋予更高排产权重。"
    )
    fig.text(0.5, 0.02, summary, ha="center", fontsize=11, color="#444444")

    legend_handles = [
        Patch(facecolor=FLAVOR_COLORS["蒸"], edgecolor="#2F4858", label="蒸菜"),
        Patch(facecolor=FLAVOR_COLORS["炒"], edgecolor="#2F4858", label="炒菜"),
        Patch(facecolor=FLAVOR_COLORS["煎"], edgecolor="#2F4858", label="煎菜"),
        Patch(facecolor="#FFFFFF", edgecolor="#B03A2E", linewidth=2, label="桌3/VIP 高权重"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0.03, 0.06, 0.98, 0.95))
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"甘特图已保存到: {save_path}")


def main():
    args = parse_args()

    if cp_model is None:
        print("未检测到 ortools，请先安装：pip install ortools")
        return

    tasks = build_tasks()
    solved = solve_schedule(tasks)
    if solved is None:
        print("未找到可行方案。")
        return

    rows = []
    for item in solved["results"]:
        rows.append(
            [
                item["table"],
                item["dish"],
                item["cooker"],
                item["start"],
                item["end"],
                item["flavor"],
                item["distance"],
                f"{item['weight']:.2f}",
            ]
        )

    rows.sort(key=lambda item: (item[3], item[2], item[0], item[1]))
    print("\n--- 餐厅后厨排产结果 ---")
    print(format_results(rows))
    print(f"\n总完工时间(Makespan): {solved['makespan']} 分钟")

    create_gantt_chart(
        solved["results"],
        solved["makespan"],
        solved["setup_time"],
        Path(args.save_path),
    )


if __name__ == "__main__":
    main()
