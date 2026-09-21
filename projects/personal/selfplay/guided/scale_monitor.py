"""Read-only Mac monitor for a finite two-host scale study.

Collects curves and paired receipts into a local report, then exits when both
host suites finish. It never launches or extends training and sends no messages.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

from guided import remote
from guided.run import atomic_json
from guided.scale_probe import PROBE_SCALES
from guided.statistics import interval


SNAPSHOT = """import pathlib,json,hashlib,time
root=pathlib.Path(ROOT)
result={'models':{},'fetched_at':time.time()}
p=root/'suite.json'
result['suite']=json.loads(p.read_text()) if p.exists() else {'stage':'not_started'}
for name in ['small','medium','large','xlarge']:
 p=root/name/'result.json'
 if not p.exists():continue
 result['models'][name]=json.loads(p.read_text())
 p=root/name/'evaluation.json'
 if p.exists():
  evaluation=json.loads(p.read_text())
  conditions=[]
  for row in evaluation['records']:
   if row['policy']!='model:current':continue
   seats=list(row['seats']);seats[row['focal']]='<focal>'
   conditions.append([row['group'],row['focal'],row['seed'],seats])
  identity=hashlib.sha256(json.dumps(sorted(conditions),separators=(',',':')).encode()).hexdigest()
  result['models'][name]['evaluation_conditions_sha256']=identity
  result['models'][name]['evaluation_receipt']=evaluation
# Reconcile recorded state with process identity after a host/session reset.
# A PID alone is insufficient because Linux may reuse it after reboot.
def command(path):
 try:return [part.decode(errors='replace') for part in path.read_bytes().split(bytes([0])) if part]
 except OSError:return []
supervisor_alive=False
for path in pathlib.Path('/proc').glob('[0-9]*/cmdline'):
 args=command(path)
 if 'guided.scale_study' in args and 'suite' in args and str(root) in args:
  supervisor_alive=True;break
result['supervisor_alive']=supervisor_alive
active=[]
for name,task in result['suite'].get('tasks',{}).items():
 if task.get('state')!='running':continue
 expected=task.get('argv',[])
 actual=command(pathlib.Path('/proc')/str(task.get('pid',-1))/'cmdline')
 alive=bool(expected) and actual==expected
 task['runtime_alive']=alive
 if alive:active.append(name);continue
 model=result['models'].get(name,{})
 task['recorded_state']=task['state']
 task['state']='complete' if model.get('stage')=='complete' else 'process_missing'
 if model and model.get('stage')!='complete':
  model['recorded_stage']=model['stage'];model['stage']='interrupted'
  model['runtime_status_reason']='recorded process is absent or its identity changed'
if result['suite'].get('stage')=='running' and not supervisor_alive:
 result['suite']['recorded_stage']='running'
 tasks=result['suite'].get('tasks',{})
 complete=set(tasks)=={'small','medium','large','xlarge'} and all(t['state']=='complete' for t in tasks.values())
 result['suite']['stage']='orphaned' if active else ('complete' if complete else 'interrupted')
print(json.dumps(result))
"""


def paired(models):
    if "medium" not in models or "evaluation_receipt" not in models["medium"]:
        return {}
    reference = models["medium"]
    teacher = reference["evaluation_receipt"]["teacher"]
    baseline = reference["evaluation_receipt"]["results"]
    output = {}
    for name, model in models.items():
        if name == "medium" or "evaluation_receipt" not in model:
            continue
        if (
            model["evaluation_conditions_sha256"]
            != reference["evaluation_conditions_sha256"]
        ):
            raise ValueError("models have different evaluation walls/seats/opponents")
        measured = model["evaluation_receipt"]["results"]
        if measured[teacher]["group_scores"] != baseline[teacher]["group_scores"]:
            raise ValueError("frozen teacher scores changed between scale conditions")
        a, b = (
            measured["model:current"]["group_scores"],
            baseline["model:current"]["group_scores"],
        )
        if len(a) != len(b):
            raise ValueError("different paired sample counts")
        output[name] = interval([x - y for x, y in zip(a, b)])
    return output


def render(plan, results, pairs, final):
    lines = [
        "# 延长模型规模对照",
        "",
        "状态："
        + (
            "两机任务均已结束，请结合失败状态及完整结果判断。"
            if final
            else "进行中，以下仅为已经记录的阶段结果。"
        ),
        "",
        f"每档纯模仿更新 {plan['seconds_per_scale'] / 60:g} 分钟；两台机器各自测试全部四档，分别使用独立种子。共同冻结数据、教师与评估条件。正式 PPO 训练未启动，没有活动的一周截止时间。",
        "",
        "更新预算不包含数据生成、初始化、阶段验证与收尾对局时间。实际墙钟耗时同时保留。各档资源微批次按先前实测调整，有效批量均为 512，FP32。",
        "",
        "| 主机 | 模型 | 阶段 | 最近更新秒数 | 验证交叉熵↓ | 教师动作一致率 | 收尾场均桌面分（95%区间） |",
        "| --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for host in ("4090", "4070"):
        for name in ("small", "medium", "large", "xlarge"):
            model = results.get(host, {}).get("models", {}).get(name, {})
            curve = model.get("curve", [])
            last = curve[-1] if curve else {}
            val = last.get("validation", {})
            game = model.get("evaluation", {}).get("results", {}).get("model:current")
            score = (
                f"{game['mean']:.2f} [{game['lower']:.2f}, {game['upper']:.2f}]"
                if game
                else "待完成"
            )
            loss = f"{val['cross_entropy']:.4f}" if val else "—"
            accuracy = f"{val['accuracy']:.2%}" if val else "—"
            dim, layers, _ = PROBE_SCALES[name]
            lines.append(
                f"| {host} | {dim}×{layers} | {model.get('stage', '等待')} | {last.get('update_seconds', 0):.1f} | {loss} | {accuracy} | {score} |"
            )
    lines += [
        "",
        "## 验证误差随更新时间变化",
        "",
        "| 主机 | 模型 | 90 秒 | 5 分钟 | 15 分钟 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for host in ("4090", "4070"):
        for name, (dim, layers, _) in PROBE_SCALES.items():
            curve = (
                results.get(host, {}).get("models", {}).get(name, {}).get("curve", [])
            )
            values = {
                point["target_update_seconds"]: point["validation"]["cross_entropy"]
                for point in curve
            }
            rendered = [
                f"{values[t]:.4f}" if t in values else "待完成" for t in (90, 300, 900)
            ]
            lines.append(f"| {host} | {dim}×{layers} | " + " | ".join(rendered) + " |")
    lines += [
        "",
        "## 相对中档的配对桌面分差",
        "",
        "正值表示该规模优于中档。区间按共享牌墙的完整四座组计算；每台机器单独比较，不能把两种 GPU 的分数直接解释成硬件导致的棋力差。",
        "",
        "| 主机 | 模型减中档 | 场均分差 | 95%配对区间 |",
        "| --- | --- | ---: | --- |",
    ]
    for host, values in pairs.items():
        for name, value in values.items():
            lines.append(
                f"| {host} | {name} − medium | {value['mean']:.2f} | [{value['lower']:.2f}, {value['upper']:.2f}] |"
            )
    lines += [
        "",
        "## 判断边界",
        "",
        "这是监督学习的规模对照及训练后实际对局测试，仍不等于长期 RL 上限。中途查看的验证集与模型比较属于开发评估；多个比较的区间没有整体多重比较保证。收尾固定使用预先约定的最终时间点模型，不按中途表现挑检查点，也不根据分数追加场次。",
        "",
        "原 90 秒对照的中档建议保持暂定，不因这里某个尚未完成的局部分数自动切换训练规模。",
        "",
        f"- 冻结训练源码：`{plan['source_id']}`",
        f"- [完整结果及逐场回执]({Path(plan['root']) / 'results.json'})",
        f"- [运行计划]({Path(plan['root']) / 'plan.json'})",
        "",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--interval", type=float, default=30)
    args = p.parse_args()
    root = Path(args.run)
    plan = json.loads((root / "plan.json").read_text())
    started = time.monotonic()
    results = {}
    with ThreadPoolExecutor(2) as pool:
        while time.monotonic() - started < 8 * 3600:

            def fetch(host):
                script = SNAPSHOT.replace("ROOT", repr(plan["remote_root"] + "/suite"))
                return host, json.loads(remote.execute(host, script))

            try:
                results = dict(pool.map(fetch, ("4090", "4070")))
                for host in results.values():
                    for model in host["models"].values():
                        if (
                            model["code_id"] != plan["source_id"]
                            or model["data_sha256"] != plan["data_sha256"]
                        ):
                            raise ValueError("source or dataset identity changed")
                pairs = {
                    host: paired(value["models"]) for host, value in results.items()
                }
                final = all(
                    value["suite"]["stage"]
                    in ("complete", "finished_with_failures", "interrupted")
                    for value in results.values()
                )
                atomic_json(root / "results.json", results)
                atomic_json(root / "paired.json", pairs)
                (root / "REPORT.md").write_text(render(plan, results, pairs, final))
                atomic_json(
                    root / "monitor.json",
                    {
                        "stage": "finished" if final else "running",
                        "pid": os.getpid(),
                        "time": time.time(),
                        "formal_training_started": False,
                        "controller_sha256": hashlib.sha256(
                            Path(__file__).read_bytes()
                        ).hexdigest(),
                    },
                )
                if final:
                    latest = json.loads((root / "plan.json").read_text())
                    latest["stage"] = (
                        "complete"
                        if all(
                            value["suite"]["stage"] == "complete"
                            for value in results.values()
                        )
                        else "finished_with_failures"
                    )
                    latest["finished"] = time.time()
                    atomic_json(root / "plan.json", latest)
                    pointer = root.parent / "current.json"
                    if pointer.exists() and json.loads(pointer.read_text()).get(
                        "root"
                    ) == str(root):
                        atomic_json(pointer, latest)
                    return
            except ValueError as exc:
                atomic_json(
                    root / "monitor.json",
                    {
                        "stage": "failed_validation",
                        "error": str(exc),
                        "time": time.time(),
                    },
                )
                raise
            except (RuntimeError, OSError, subprocess.TimeoutExpired) as exc:
                atomic_json(
                    root / "monitor-error.json",
                    {"time": time.time(), "error": str(exc)},
                )
            time.sleep(args.interval)
    raise TimeoutError("monitor window expired; inspect finite remote suite status")


if __name__ == "__main__":
    main()
