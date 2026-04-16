import numpy as np
import napari

# ==========================
# 1. Load the 3D volume
# ==========================
volume_path = "/home/huangxn/Desktop/Playground/virtual_embryo/heem/preimp_mouse/early_mouse/dataset/resized_img.npy"
#volume_path = "/home/huangxn/Downloads/T1_010.npz"
#data = np.load(volume_path)
volume = np.load(volume_path)

print(f"Loaded volume shape (Z, Y, X): {volume.shape}")

# ==========================
# 2. Define physical voxel scaling (essential for correct embryo proportions)
# ==========================
dz = 2.0    # Z spacing (μm)
dy = 0.208  # Y spacing (μm)
dx = 0.208  # X spacing (μm)

scale = (dz, dy, dx)   # Order must match data: (Z, Y, X)

# ==========================
# 3. Create napari viewer and add the image layer
# ==========================
viewer = napari.Viewer(title="BlastoSPIM - T1_010_image_0001 (3D Volume with Z Slider)")

layer = viewer.add_image(
    volume,
    name="Embryo Volume",
    colormap="magma",                    # Suitable for nuclear fluorescence; try "magma" for higher contrast
    contrast_limits=[volume.min(), volume.max()],
    scale=scale,                        # Corrects for anisotropy
    blending="additive",
    rendering="mip",                    # Maximum Intensity Projection – excellent for dense nuclei
    interpolation2d="nearest",          # For 2D slicing
    interpolation3d="linear"           # For 3D rendering (or "nearest" for sharper voxels)
)

# Optional enhancements for 3D rendering
layer.rendering = "iso"      # Improves depth perception in thick volumes
layer.iso_threshold = 0.1             # Uncomment and tune if using 'iso' rendering

# ==========================
# 4. Configure dimension slider and initial view
# ==========================
viewer.dims.axis_labels = ['Z (slices)', 'Y', 'X']
viewer.dims.ndisplay = 2                # Start in 3D volumetric mode
                                        # Change to 2 to start in 2D slicing mode with Z slider prominent

# ==========================
# 5. Launch the interactive viewer
# ==========================
napari.run()