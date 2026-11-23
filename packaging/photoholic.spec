# PyInstaller spec for Photoholic.exe (one-folder, windowed). Build with packaging\build.ps1.
# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

ROOT = Path(SPECPATH).parent  # noqa: F821 - SPECPATH is provided by PyInstaller
ENGINE = ROOT / "image-reader" / "target" / "release" / "image-reader.exe"
MODEL = ROOT / "ai" / "models" / "aesthetic_v1.onnx"
ICON = ROOT / "app" / "photoholic.ico"

for required in (ENGINE, MODEL, MODEL.with_suffix(".json"), ICON):
    if not required.is_file():
        raise SystemExit(f"missing {required} - see packaging/build.ps1")

a = Analysis(  # noqa: F821
    [str(ROOT / "app" / "launcher.py")],
    pathex=[str(ROOT / "ai"), str(ROOT / "app")],
    binaries=[(str(ENGINE), "engine")],
    datas=[
        (str(ROOT / "app" / "static"), "app/static"),
        (str(ICON), "app"),
        (str(MODEL), "models"),
        (str(MODEL.with_suffix(".json")), "models"),
    ],
    hiddenimports=[],
    excludes=[
        # training-only or optional (charts, agglomerative grouping): keeps the app small
        "torch", "torchvision", "training", "onnx", "onnxscript",
        "matplotlib", "sklearn", "scipy", "pytest", "IPython", "notebook",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure)  # noqa: F821
exe = EXE(  # noqa: F821
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Photoholic",
    icon=str(ICON),
    console=False,  # windowed: no black console window
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name="Photoholic", upx=False)  # noqa: F821
