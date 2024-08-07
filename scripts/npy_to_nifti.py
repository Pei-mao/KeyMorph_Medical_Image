import numpy as np
import nibabel as nib
import sys

def read_affine_from_nifti(nifti_file):
    nifti_img = nib.load(nifti_file)
    return nifti_img.affine

def swap_dimensions(data):
    # 互換第一個和第四個維度
    if data.ndim == 4:
        data = data.transpose(3, 1, 2, 0)
    else:
        print("Error: The input data does not have 4 dimensions.")
        sys.exit(1)
    return data

def rescale_affine(affine, scale_factor):
    # 創建一個單位矩陣
    rescaled_affine = np.eye(4)
    print("reference: ", affine)
    # 對前 3x3 的子矩陣進行縮放
    #rescaled_affine[:3, :3] = affine[:3, :3] * scale_factor
    rescaled_affine[:3, :3] = affine[:3, :3]
    rescaled_affine[0, 0] = rescaled_affine[0, 0] * 0.5
    rescaled_affine[1, 1] = rescaled_affine[1, 1] * 0.5
    rescaled_affine[2, 2] = rescaled_affine[2, 2] * 0.5
    
    # 保留平移部分
    rescaled_affine[:3, 3] = affine[:3, 3]
    return rescaled_affine

def npy_to_nifti_with_affine(npy_file, reference_nifti_file, output_nifti_file, scale_factor):
    # 讀取 .npy 文件
    data = np.load(npy_file)
    
    # 讀取參考 NIfTI 文件中的仿射矩陣
    affine = read_affine_from_nifti(reference_nifti_file)
    
    # 縮放仿射矩陣
    affine = rescale_affine(affine, scale_factor)
    
    # 互換第一個和第四個維度
    data = swap_dimensions(data)
    
    # 創建 NIfTI 圖像
    nifti_img = nib.Nifti1Image(data, affine)
    print("output: ", affine)
    # 保存為 .nii 文件
    nib.save(nifti_img, output_nifti_file)
    print(f"Saved NIfTI file to {output_nifti_file}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python npy_to_nifti.py <input.npy> <reference_nifti.nii> <output.nii> <scale_factor>")
        sys.exit(1)
    
    npy_file = sys.argv[1]
    reference_nifti_file = sys.argv[2]
    output_nifti_file = sys.argv[3]
    scale_factor = float(sys.argv[4])
    
    npy_to_nifti_with_affine(npy_file, reference_nifti_file, output_nifti_file, scale_factor)
