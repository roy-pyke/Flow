"""Create a self-contained local milestone folder without copying Git/node caches.

The optional environment copy is for this same computer: Python venvs are not
portable to another operating system. Other computers should run setup.sh.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination",type=Path)
    parser.add_argument("--include-environment",action="store_true")
    args=parser.parse_args()
    destination=args.destination.expanduser().resolve()
    if destination.exists(): raise SystemExit(f"Destination already exists; choose a new folder: {destination}")
    if destination==ROOT or ROOT in destination.parents:
        raise SystemExit("Choose a sibling or external folder, outside the source tree.")
    ignored={".git","node_modules","__pycache__",".pytest_cache",".playwright-cli",".DS_Store","output","cache",".env"}
    if not args.include_environment:ignored.add(".venv")
    shutil.copytree(ROOT,destination,symlinks=True,ignore=lambda _path,names:[n for n in names if n in ignored or n.startswith('.env.')])
    commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
    info={"created_at":datetime.now(timezone.utc).isoformat(),"git_commit":commit,
          "source":str(ROOT),"environment_included":args.include_environment,
          "note":"Environment copy is usable on this machine; on another machine rerun setup.sh. Git history and node_modules are omitted; production build and experiment history are included."}
    (destination/"ARCHIVE_INFO.json").write_text(json.dumps(info,indent=2)+"\n")
    (destination/"从这里开始.md").write_text(f'''# Flow · Plan 1 阶段归档

保存时间：{info['created_at']}。对应代码提交：`{commit}`。

这个独立文件夹保存了完整源代码、真实地图数据、编译好的界面、实验历史、验证报告、演示视频和进度记录。

- 进度：打开 `artifacts/plan1-2026-09-17/PROGRESS.md`。
- 演示：打开 `artifacts/plan1-2026-09-17/demo.webm`。
- 当前电脑运行：双击 `Start Flow.command`，浏览器打开 http://127.0.0.1:8000 。
- 如果原项目还在运行，先关闭原项目的服务，或运行 `FLOW_PORT=8001 ./start.sh`。
- 换电脑：依赖需要重新安装，运行 `./setup.sh`，再运行 `./start.sh`。
- 原项目继续开发的位置：`{ROOT}`。

`.git`、前端开发依赖和临时浏览器文件没有复制。完整生产界面已包含；快照文件哈希保存在 `SNAPSHOT_SHA256.json`。
''')
    files={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in destination.rglob("*") if p.is_file() and ".venv" not in p.parts}
    (destination/"SNAPSHOT_SHA256.json").write_text(json.dumps(files,indent=2)+"\n")
    print(json.dumps({"destination":str(destination),"git_commit":commit,"files":len(files)},indent=2))


if __name__=="__main__":main()
