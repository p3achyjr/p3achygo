"""
Migrates old .keras checkpoints to use p3achygo> registered names.

Patches config.json inside the ZIP in-place (backs up original first).

Usage:
    python .claude_scripts/migrate_keras_checkpoint.py /tmp/scratch.keras
    python .claude_scripts/migrate_keras_checkpoint.py /path/to/models/*.keras
"""
import argparse
import glob
import json
import os
import shutil
import zipfile

# Map old registered_name -> new registered_name
MIGRATIONS = {
    # P3achyGoModel was saved with no prefix before decorators were added
    "P3achyGoModel": "p3achygo>P3achyGoModel",
    # Optimizer + LR schedules had wrong packages
    "p3achygo>ConvMuon": "p3achygo>ConvMuon",          # already correct, no-op
    "custom>ConstantLRSchedule": "p3achygo>ConstantLRSchedule",
    "custom>ConvSWS": "p3achygo>ConvSWS",
    # Layer classes that had no decorator at all (registered_name was None or class name)
    "ConvBlock": "p3achygo>ConvBlock",
    "ConvPostActivation": "p3achygo>ConvPostActivation",
    "ConvPreActivation": "p3achygo>ConvPreActivation",
    "ResidualBlock": "p3achygo>ResidualBlock",
    "ClassicResidualBlock": "p3achygo>ClassicResidualBlock",
    "BottleneckResidualConvBlock": "p3achygo>BottleneckResidualConvBlock",
    "NbtResidualBlock": "p3achygo>NbtResidualBlock",
    "BroadcastResidualBlock": "p3achygo>BroadcastResidualBlock",
    "Broadcast": "p3achygo>Broadcast",
    "BroadcastPostAct": "p3achygo>BroadcastPostAct",
    "BroadcastPreAct": "p3achygo>BroadcastPreAct",
    "GlobalPool": "p3achygo>GlobalPool",
    "GlobalPoolBias": "p3achygo>GlobalPoolBias",
    "PolicyHead": "p3achygo>PolicyHead",
    "ValueHead": "p3achygo>ValueHead",
}


def patch_config(obj):
    """Recursively walk the config dict and update registered_name fields."""
    if isinstance(obj, dict):
        if "registered_name" in obj and obj["registered_name"] in MIGRATIONS:
            old = obj["registered_name"]
            new = MIGRATIONS[old]
            if old != new:
                print(f"    {old!r} -> {new!r}")
            obj["registered_name"] = new
        for v in obj.values():
            patch_config(v)
    elif isinstance(obj, list):
        for item in obj:
            patch_config(item)


def migrate(path: str, dry_run: bool = False):
    print(f"\nMigrating: {path}")
    backup = path + ".bak"
    if not os.path.exists(backup):
        shutil.copy2(path, backup)
        print(f"  Backed up to {backup}")
    else:
        print(f"  Backup already exists: {backup}")

    with zipfile.ZipFile(path, "r") as zin:
        names = zin.namelist()
        if "config.json" not in names:
            print("  No config.json found — skipping.")
            return
        config = json.loads(zin.read("config.json"))
        other_files = {n: zin.read(n) for n in names if n != "config.json"}

    print("  Patching registered_name fields:")
    patch_config(config)

    if dry_run:
        print("  [dry-run] Would write patched config.json")
        return

    tmp_path = path + ".tmp"
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        zout.writestr("config.json", json.dumps(config))
        for name, data in other_files.items():
            zout.writestr(name, data)

    os.replace(tmp_path, path)
    print(f"  Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help=".keras file(s) to migrate (globs ok)")
    parser.add_argument("--dry-run", action="store_true", help="Show changes without writing")
    args = parser.parse_args()

    expanded = []
    for p in args.paths:
        expanded.extend(glob.glob(p))
    if not expanded:
        print("No files matched.")
        return

    for path in expanded:
        migrate(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
