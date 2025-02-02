#%%
import os
import cv2
#%%
from src.descriptor_utils import extract_features
from src.matching_utils import match_features, compute_match_metrics
from src.visualization import *
#%%
pairs_dir = "/Users/gorazdfilipovski/PycharmProjects/LocalDescriptorsComparison/data/pairs"  
pair_names = ["pair1", "pair2", "pair3", "pair4","pair5","pair6"]  
#%%
descriptor_methods = ["SIFT", "ORB", "BRIEF", "BRISK", "AKAZE", "KAZE", "ROOTSIFT"]
#%%
all_results = []
for pair_name in pair_names:
    img1_path = os.path.join(pairs_dir, pair_name, "image1.jpg")
    img2_path = os.path.join(pairs_dir, pair_name, "image2.jpg")
    
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)
    
    if img1 is None or img2 is None:
        print(f"Warning: Could not read images in {pair_name}")
        continue
    
    print(f"\n--- Processing {pair_name} ---")
    
    for method in descriptor_methods:
        print(f"Descriptor: {method}")
        
        kp1, desc1 = extract_features(img1, method=method)
        kp2, desc2 = extract_features(img2, method=method)
        
        matches = match_features(desc1, desc2, method="bf", ratio_test=True)
        
        metrics = compute_match_metrics(kp1, kp2, matches)
        print(metrics)
        
        # draw_matches(img1, kp1, img2, kp2, matches, max_matches=50)
        # draw_thick_matches(img1, kp1, img2, kp2, matches, max_matches=50, line_thickness=3)
        draw_colored_matches(img1, kp1, img2, kp2, matches, max_matches=50, line_thickness=3)
        
        result_entry = {
            "pair": pair_name,
            "descriptor": method,
            "num_keypoints1": metrics["num_keypoints_image1"],
            "num_keypoints2": metrics["num_keypoints_image2"],
            "num_good_matches": metrics["num_good_matches"]
        }
        all_results.append(result_entry)

#%%
import pandas as pd

df_results = pd.DataFrame(all_results)
display(df_results)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.barplot(data=df_results, x='descriptor', y='num_keypoints1', hue='pair')
plt.title("Number of Keypoints (Image 1) by Descriptor and Pair")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.barplot(data=df_results, x='descriptor', y='num_good_matches', hue='pair')
plt.title("Number of Good Matches by Descriptor and Pair")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
#%%
pivot_table = df_results.pivot(index='pair', columns='descriptor', values='num_good_matches')
print(pivot_table)
#%%
import seaborn as sns
import matplotlib.pyplot as plt

pivot_matches = df_results.pivot(
    index='pair',
    columns='descriptor',
    values='num_good_matches'
)
plt.figure(figsize=(8,6))
sns.heatmap(pivot_matches, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Heatmap of Good Matches")
plt.show()
#%%
pivot_matches = df_results.pivot(
    index='pair',
    columns='descriptor',
    values='num_good_matches'
)
pivot_matches.plot(kind='line', figsize=(8,6), marker='o')
plt.title("Good Matches per Pair (Line Plot by Descriptor)")
plt.ylabel("Number of Good Matches")
plt.show()
#%%
sns.scatterplot(
    data=df_results, 
    x='num_keypoints1', 
    y='num_good_matches', 
    hue='descriptor', 
    style='pair'
)
plt.title("Keypoints vs. Good Matches by Descriptor and Pair")
plt.show()
#%%
sns.boxplot(x='descriptor', y='num_good_matches', data=df_results)
plt.title("Distribution of Good Matches by Descriptor")
plt.xticks(rotation=45)
plt.show()
#%%
