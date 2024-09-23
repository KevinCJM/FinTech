# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(['/Users/chenjunming/Desktop/python_algorithm/KYP/test_2.py'],
             pathex=['/Users/chenjunming/Desktop'],
             binaries=[],
             datas=[],
             hiddenimports=[
                 'PyQt5.QtCore',
                 'PyQt5.QtGui',
                 'PyQt5.QtWidgets',
                 'numpy.core._methods',
                 'numpy.lib.format'
             ],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data,
          cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='ScreenToGIF',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False )

app = BUNDLE(exe,
             name='ScreenToGIF.app',
             icon='/Users/chenjunming/Desktop/gif_29865.icns',
             bundle_identifier=None)
