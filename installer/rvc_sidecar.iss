; Vox Studio Pass 9a RVC sidecar payload.
; Included from installer/voxstudio.iss once the full installer pass is active.

#define RvcSidecarSource "..\third_party\rvc_sidecar"

[Files]
Source: "{#RvcSidecarSource}\*"; DestDir: "{localappdata}\VoxStudio\rvc_sidecar"; \
    Flags: recursesubdirs createallsubdirs ignoreversion

[UninstallDelete]
Type: filesandordirs; Name: "{localappdata}\VoxStudio\rvc_sidecar"
