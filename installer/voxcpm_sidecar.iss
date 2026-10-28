; Vox Studio VoxCPM2 phrase-live sidecar payload.

#define VoxCpmSidecarSource "..\third_party\voxcpm_sidecar"

[Files]
Source: "{#VoxCpmSidecarSource}\*"; DestDir: "{localappdata}\VoxStudio\voxcpm_sidecar"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\VoxStudio\voxcpm_sidecar"
