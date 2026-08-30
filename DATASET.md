# Copy to a staging folder, excluding junk
 $src = ""C:\Users\diwak\Desktop\MVP-rig_dataset""
 $stage = "C:\Users\diwak\Desktop\MVP-Rig-Dataset"
robocopy $src $stage /E /XD bone_3d_backup_mirrored pose_fix_backup bone_3d_original_backup /XF *.fbx *.mtl

# Zip it
Compress-Archive -Path $stage -DestinationPath "$stage.zip"

# Checksum (reproducibility practice)
Get-FileHash "$stage.zip" -Algorithm SHA256