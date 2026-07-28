; Vox Studio Performance Mirror sidecar payload.

#define PerformanceMirrorSource "..\third_party\performance_mirror_sidecar"

[Files]
Source: "{#PerformanceMirrorSource}\*"; \
    DestDir: "{localappdata}\VoxStudio\performance_mirror_sidecar"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\VoxStudio\performance_mirror_sidecar"
