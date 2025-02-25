import os
import argparse
import nibabel as nib
import numpy as np
import csv
import tempfile

def parse_arguments():
    parser = argparse.ArgumentParser(description="Calculate Dice score for a label in two segmented images.")
    parser.add_argument("-input1", required=True, help="Absolute path to the first segmented file")
    parser.add_argument("-input2", required=True, help="Absolute path to the second segmented file")
    parser.add_argument("-label", type=int, required=True, help="Label number to compare volumes")
    parser.add_argument("-temp", default=tempfile.gettempdir(), help="Absolute path for temporary directory (default: system temp directory)")
    return parser.parse_args()

def check_files_exist(filepath):
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ERROR: {filepath} does not exist. Check path and filename!")

def ensure_temp_directory(temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)

def compute_dice(input1, input2, label):
    img1 = nib.load(input1).get_fdata()
    img2 = nib.load(input2).get_fdata()
    
    if img1.size == 0 or img2.size == 0:
        raise ValueError("One or both input images are empty.")
    
    if img1.shape != img2.shape:
        raise ValueError("Input images have mismatched dimensions.")
    
    label_mask1 = (img1 == label)
    label_mask2 = (img2 == label)
    
    intersection = np.logical_and(label_mask1, label_mask2).sum()
    vol1 = label_mask1.sum()
    vol2 = label_mask2.sum()
    
    if vol1 == 0 or vol2 == 0:
        dice_score = 0.0
    else:
        dice_score = (2.0 * intersection) / (vol1 + vol2)
    
    return dice_score, vol1, vol2, intersection

def main():
    args = parse_arguments()
    
    check_files_exist(args.input1)
    check_files_exist(args.input2)
    ensure_temp_directory(args.temp)
    
    dice_score, vol1, vol2, commonvol = compute_dice(args.input1, args.input2, args.label)
    
    print(f"\nVolume for label {args.label} in input1 = {vol1}")
    print(f"Volume for label {args.label} in input2 = {vol2}")
    print(f"Area of overlapped volumes = {commonvol}")
    print(f"\nDice score for label {args.label} is {dice_score}\n")
    
    csv_path = os.path.join(args.temp, "dice_scores.csv")
    with open(csv_path, mode='a', newline='') as csv_file:
        writer = csv.writer(csv_file)
        if os.stat(csv_path).st_size == 0:
            writer.writerow(["Label", "Volume Input1", "Volume Input2", "Common Volume", "Dice Score"])
        writer.writerow([args.label, vol1, vol2, commonvol, dice_score])

if __name__ == "__main__":
    main()

