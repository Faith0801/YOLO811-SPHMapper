YOLO811-SPHMapper Pipeline Implementation
Complete Python implementation of the three-stage PV panel inspection pipeline described in the master's thesis "Research on Automated Detection and Assessment of Photovoltaic Panel Surface Conditions Using a Multi-Model Deep Learning Pipeline", North China Electric Power University.
Pipeline stages:

Stage 1: Panel surface segmentation (YOLOv8m-Seg)
Stage 2: Anomaly detection — cracks and bird droppings (YOLOv8m-Det)
Stage 3: Tile-level dust severity classification (YOLOv11-Cls)

Output: Colour-coded dust severity overlay and dual-layer pie chart dashboard per panel, compiled into a PDF report.
Requirements: Python 3.8+, ultralytics, opencv-python, numpy, matplotlib, scipy, tqdm, reportlab
Model weights (.pt files): Available on request from the author.
