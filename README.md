# Cyberomix_analysis
株式会社Cyberomixで用いた解析例
(守秘義務に基づき、一般公開されているturtorialデータを元に、readmeスクリプト上でのみ許可されましたので、共有します。)

## Nuclei Segmentation and Custom Binning of Visium HD Gene Expression Dataにおける解析(python)

# はじめに
ここでは、HE染色画像における位置情報と遺伝子発現の位置情報を組み合わせ、特定の遺伝子が発現している細胞のマッピングを行うことができる解析である、Nuclei Segmentation and Custom Binning of Visium HD Gene Expression Dataにおける解析について述べる。

# installとバージョンの確認
① pythonディレクトリのinstall

```
# condaのアップデート
conda update conda

# condaの環境構築
conda create -n stardist-env
conda activate stardist-env
conda config --env --add channels conda-forge
conda config --env --set channel_priority strict

# Install Python libraries
conda install -c conda-forge stardist
conda install python=3 geopandas
conda install -c conda-forge squidpy
conda install -c conda-forge fastparquet
```

# turtorialにおけるAnalysis data download
① outputs dataのdownload
jupyter notebookにて、以下のコマンドでdownload。この時downloadされるファイルはgzipファイルなので、ファイルの解凍も行う必要がある。

```
# ファイルのdownload
!curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz

#gzipファイルの解凍
!tar -xzvf Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
```

② high-resolution H&E microscope datasetのdownload
jupyter notebookにて、以下のコマンドでdownload。

```
# ファイルのdownload
!curl -O https://cf.10xgenomics.com/samples/spatial-exp/3.0.0/Visium_HD_Mouse_Small_Intestine/Visium_HD_Mouse_Small_Intestine_binned_outputs.tar.gz
```

# libraryのimport
python libraryのimport及びjupyter notebookの画像プロットの設定

```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import anndata
import geopandas as gpd
import scanpy as sc

from tifffile import imread, imwrite
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from shapely.geometry import Polygon, Point
from scipy import sparse
from matplotlib.colors import ListedColormap

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
```

# プロット関数の定義
結果をplotする際に必要な関数の定義

```
# General image plotting functions
def plot_mask_and_save_image(title, gdf, img, cmap, output_name=None, bbox=None):
    if bbox is not None:
        # Crop the image to the bounding box
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # Plot options
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the cropped image
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    # Create filtering polygon
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])
        # Filter for polygons in the box
        intersects_bbox = gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = gdf[intersects_bbox]
    else:
        filtered_gdf=gdf

    # Plot the filtered polygons on the second axis
    filtered_gdf.plot(cmap=cmap, ax=axes[1])
    axes[1].axis('off')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))


    # Save the plot if output_name is provided
    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend
    else:
        plt.show()

def plot_gene_and_save_image(title, gdf, gene, img, adata, bbox=None, output_name=None):

    if bbox is not None:
        # Crop the image to the bounding box
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    # Plot options
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the cropped image
    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    # Create filtering polygon
    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])


    # Find a gene of interest and merge with the geodataframe
    gene_expression = adata[:, gene].to_df()
    gene_expression['id'] = gene_expression.index
    merged_gdf = gdf.merge(gene_expression, left_on='id', right_on='id')

    if bbox is not None:
        # Filter for polygons in the box
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # Plot the filtered polygons on the second axis
    filtered_gdf.plot(column=gene, cmap='inferno', legend=True, ax=axes[1])
    axes[1].set_title(gene)
    axes[1].axis('off')
    axes[1].legend(loc='upper left', bbox_to_anchor=(1.05, 1))

    # Save the plot if output_name is provided
    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')  # Use bbox_inches='tight' to include the legend
    else:
        plt.show()

def plot_clusters_and_save_image(title, gdf, img, adata, bbox=None, color_by_obs=None, output_name=None, color_list=None):
    color_list=["#7f0000","#808000","#483d8b","#008000","#bc8f8f","#008b8b","#4682b4","#000080","#d2691e","#9acd32","#8fbc8f","#800080","#b03060","#ff4500","#ffa500","#ffff00","#00ff00","#8a2be2","#00ff7f","#dc143c","#00ffff","#0000ff","#ff00ff","#1e90ff","#f0e68c","#90ee90","#add8e6","#ff1493","#7b68ee","#ee82ee"]
    if bbox is not None:
        cropped_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        cropped_img = img

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(cropped_img, cmap='gray', origin='lower')
    axes[0].set_title(title)
    axes[0].axis('off')

    if bbox is not None:
        bbox_polygon = Polygon([(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])])

    unique_values = adata.obs[color_by_obs].astype('category').cat.categories
    num_categories = len(unique_values)

    if color_list is not None and len(color_list) >= num_categories:
        custom_cmap = ListedColormap(color_list[:num_categories], name='custom_cmap')
    else:
        # Use default tab20 colors if color_list is insufficient
        tab20_colors = plt.cm.tab20.colors[:num_categories]
        custom_cmap = ListedColormap(tab20_colors, name='custom_tab20_cmap')

    merged_gdf = gdf.merge(adata.obs[color_by_obs].astype('category'), left_on='id', right_index=True)

    if bbox is not None:
        intersects_bbox = merged_gdf['geometry'].intersects(bbox_polygon)
        filtered_gdf = merged_gdf[intersects_bbox]
    else:
        filtered_gdf = merged_gdf

    # Plot the filtered polygons on the second axis
    plot = filtered_gdf.plot(column=color_by_obs, cmap=custom_cmap, ax=axes[1], legend=True)
    axes[1].set_title(color_by_obs)
    legend = axes[1].get_legend()
    legend.set_bbox_to_anchor((1.05, 1))
    axes[1].axis('off')

    # Move legend outside the plot
    plot.get_legend().set_bbox_to_anchor((1.25, 1))

    if output_name is not None:
        plt.savefig(output_name, bbox_inches='tight')
    else:
        plt.show()

# Plotting function for nuclei area distribution
def plot_nuclei_area(gdf,area_cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    # Plot the histograms
    axs[0].hist(gdf['area'], bins=50, edgecolor='black')
    axs[0].set_title('Nuclei Area')

    axs[1].hist(gdf[gdf['area'] < area_cut_off]['area'], bins=50, edgecolor='black')
    axs[1].set_title('Nuclei Area Filtered:'+str(area_cut_off))

    plt.tight_layout()
    plt.show()

# Total UMI distribution plotting function
def total_umi(adata_, cut_off):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Box plot
    axs[0].boxplot(adata_.obs["total_counts"], vert=False, widths=0.7, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    axs[0].set_title('Total Counts')

    # Box plot after filtering
    axs[1].boxplot(adata_.obs["total_counts"][adata_.obs["total_counts"] > cut_off], vert=False, widths=0.7, patch_artist=True, boxprops=dict(facecolor='skyblue'))
    axs[1].set_title('Total Counts > ' + str(cut_off))

    # Remove y-axis ticks and labels
    for ax in axs:
        ax.get_yaxis().set_visible(False)

    plt.tight_layout()
    plt.show()
```
# 細胞核のマスクとジオデータフレームの作成
① 画像をimportした後、画像をpercentile正規化する。min_percentile と max_percentileは必要に応じて調整をする。

```
#  image fileのload
#  /path_to_data/にbtfファイルをdownloadしたdirectoryを入力

dir_base = '/path_to_data/'
filename = 'Visium_HD_Mouse_Small_Intestine_tissue_image.btf'
img = imread(dir_base + filename)

# 学習されたモデルのload
model = StarDist2D.from_pretrained('2D_versatile_he')

# Percentile normalization of the image
# Adjust min_percentile と max_percentileは必要に応じて調整
min_percentile = 5
max_percentile = 95
img = normalize(img, min_percentile, max_percentile)
```
② 核セグメンテーションのマスク
nms_threshは核が重なる確率を減らすために小さな数値に設定する。
prob_threshを大きくするとsegmentationされる核の数が少なくなる。
これらの最適値は核segmentationマスクの目視評価によって決定。

```
# Predict cell nuclei using the normalized image
# Adjust nms_thresh and prob_thresh as needed

labels, polys = model.predict_instances_big(img, axes='YXC', block_size=4096, prob_thresh=0.01,nms_thresh=0.001, min_overlap=128, context=128, normalizer=None, n_tiles=(4,4,1))
```
③ StarDistの結果をGeodataframeへ変換
```
# Creating a list to store Polygon geometries
geometries = []

# Iterating through each nuclei in the 'polys' DataFrame
for nuclei in range(len(polys['coord'])):

    # Extracting coordinates for the current nuclei and converting them to (y, x) format
    coords = [(y, x) for x, y in zip(polys['coord'][nuclei][0], polys['coord'][nuclei][1])]

    # Creating a Polygon geometry from the coordinates
    geometries.append(Polygon(coords))

# Creating a GeoDataFrame using the Polygon geometries
gdf = gpd.GeoDataFrame(geometry=geometries)
gdf['id'] = [f"ID_{i+1}" for i, _ in enumerate(gdf.index)]
```
# 注目領域の範囲設定
Fijiを用いて、注目する領域の(x min,y min, x max,y max)を測定。Fijiの詳細な使い方は、[こちら](https://www.10xgenomics.com/jp/analysis-guides/segmentation-visium-hd)を参照。

# Cell segmentation
```
# Plot the nuclei segmentation
# bbox=(x min,y min,x max,y max)

# Define a single color cmap
cmap=ListedColormap(['grey'])

# Create Plot
# Fijiを用いて測定した(x min,y min, x max,y max)をbbox=(x min,y min,x max,y max)に代入
plot_mask_and_save_image(title="Region of Interest 1",gdf=gdf,bbox=(x min,y min,x max,y max),cmap=cmap,img=img,output_name=dir_base+"image_mask.ROI1.tif")
```

<img src="https://aw-lab.notepm.jp/private/1cbb0862-42a3-11ef-a460-06720a606bea.png" title="Region of Interest_mask" width="640">


# Binning Visium HD Gene expression Data
①遺伝子発現データと組織の位置データの読み込み
matrix.h5ファイルは、ダウンロードし、解凍されたgzipファイル内にあるファイルを指定済み。
tissue_positions.parquetファイルは、ダウンロードされたファイルを指定済み。
他のデータを用いる際は、dir_base以降のファイルpassを指定する。

```
# Load Visium HD data
# ''に具体的なpassを代入。
raw_h5_file = dir_base+'binned_outputs/square_002um/filtered_feature_bc_matrix.h5'
adata = sc.read_10x_h5(raw_h5_file)

# Load the Spatial Coordinates
# ''に具体的なpassを代入
tissue_position_file = dir_base+'binned_outputs/square_002um/spatial/tissue_positions.parquet'
df_tissue_positions=pd.read_parquet(tissue_position_file)

#Set the index of the dataframe to the barcodes
df_tissue_positions = df_tissue_positions.set_index('barcode')

# Create an index in the dataframe to check joins
df_tissue_positions['index']=df_tissue_positions.index

# Adding the tissue positions to the meta data
adata.obs =  pd.merge(adata.obs, df_tissue_positions, left_index=True, right_index=True)

# Create a GeoDataFrame from the DataFrame of coordinates
geometry = [Point(xy) for xy in zip(df_tissue_positions['pxl_col_in_fullres'], df_tissue_positions['pxl_row_in_fullres'])]
gdf_coordinates = gpd.GeoDataFrame(df_tissue_positions, geometry=geometry)
```
②核segmentation重複のチェック
各バーコードが細胞内にあるかをチェックし、一意に割り当てられたバーコードのみを保持するフィルターをかける。(重複を防ぐため)

```
# Perform a spatial join to check which coordinates are in a cell nucleus
result_spatial_join = gpd.sjoin(gdf_coordinates, gdf, how='left', predicate='within')

# Identify nuclei associated barcodes and find barcodes that are in more than one nucleus
result_spatial_join['is_within_polygon'] = ~result_spatial_join['index_right'].isna()
barcodes_in_overlaping_polygons = pd.unique(result_spatial_join[result_spatial_join.duplicated(subset=['index'])]['index'])
result_spatial_join['is_not_in_an_polygon_overlap'] = ~result_spatial_join['index'].isin(barcodes_in_overlaping_polygons)

# Remove barcodes in overlapping nuclei
barcodes_in_one_polygon = result_spatial_join[result_spatial_join['is_within_polygon'] & result_spatial_join['is_not_in_an_polygon_overlap']]

# The AnnData object is filtered to only contain the barcodes that are in non-overlapping polygon regions
filtered_obs_mask = adata.obs_names.isin(barcodes_in_one_polygon['index'])
filtered_adata = adata[filtered_obs_mask,:]

# Add the results of the point spatial join to the Anndata object
filtered_adata.obs =  pd.merge(filtered_adata.obs, barcodes_in_one_polygon[['index','geometry','id','is_within_polygon','is_not_in_an_polygon_overlap']], left_index=True, right_index=True)
```
③遺伝子毎の細胞count summation

```
# Group the data by unique nucleous IDs
groupby_object = filtered_adata.obs.groupby(['id'], observed=True)

# Extract the gene expression counts from the AnnData object
counts = filtered_adata.X

# Obtain the number of unique nuclei and the number of genes in the expression data
N_groups = groupby_object.ngroups
N_genes = counts.shape[1]

# Initialize a sparse matrix to store the summed gene counts for each nucleus
summed_counts = sparse.lil_matrix((N_groups, N_genes))

# Lists to store the IDs of polygons and the current row index
polygon_id = []
row = 0

# Iterate over each unique polygon to calculate the sum of gene counts.
for polygons, idx_ in groupby_object.indices.items():
    summed_counts[row] = counts[idx_].sum(0)
    row += 1
    polygon_id.append(polygons)

# Create and AnnData object from the summed count matrix
summed_counts = summed_counts.tocsr()
grouped_filtered_adata = anndata.AnnData(X=summed_counts,obs=pd.DataFrame(polygon_id,columns=['id'],index=polygon_id),var=filtered_adata.var)

%store grouped_filtered_adata
```

# 結果の可視化
不適切にsegmentationされた核やUMI値が低い核にフィルターをかけることで、クラスターの解釈や可視化の向上につながる。これはオプションであり、これを実行するかどうかはサンプルに依存する。

①フィルタリング
turtorialでは、核分布に基づき、area_cut_off=500でフィルタリング

```
# Store the area of each nucleus in the GeoDataframe
gdf['area'] = gdf['geometry'].area

# Calculate quality control metrics for the original AnnData object
sc.pp.calculate_qc_metrics(grouped_filtered_adata, inplace=True)

# Plot the nuclei area distribution before and after filtering
plot_nuclei_area(gdf=gdf,area_cut_off=500)
```
<img src="https://aw-lab.notepm.jp/private/7b95eb90-42a3-11ef-97c8-064017533d40.png" title="nuclei filtering plot" width="640">


② UMI分布の可視化
```
# Plot total UMI distribution
total_umi(grouped_filtered_adata, 100)
```

<img src="https://aw-lab.notepm.jp/private/9ea77afe-42a3-11ef-9f4e-064017533d40.png" title="total UMI distribution plot" width="640">


ここでは、総UMIカットオフ値を100としている。カットオフ値の選択は、クラスタリング結果に基づいて調整する必要ある。

③データのクラスタリング
```
# Create a mask based on the 'id' column for values present in 'gdf' with 'area' less than 500
mask_area = grouped_filtered_adata.obs['id'].isin(gdf[gdf['area'] < 500].id)

# Create a mask based on the 'total_counts' column for values greater than 100
mask_count = grouped_filtered_adata.obs['total_counts'] > 100

# Apply both masks to the original AnnData to create a new filtered AnnData object
count_area_filtered_adata = grouped_filtered_adata[mask_area & mask_count, :]

# Calculate quality control metrics for the filtered AnnData object
sc.pp.calculate_qc_metrics(count_area_filtered_adata, inplace=True)
```
④クラスタリング結果の評価
クラスタリング結果の評価を行うために、遺伝子発現とクラスターを調べる。
Leidenのresolutionパラメータは異なるデータセットを用いる際には調整が必要である。

```
# Normalize total counts for each cell in the AnnData object
sc.pp.normalize_total(count_area_filtered_adata, inplace=True)

# Logarithmize the values in the AnnData object after normalization
sc.pp.log1p(count_area_filtered_adata)

# Identify highly variable genes in the dataset using the Seurat method
sc.pp.highly_variable_genes(count_area_filtered_adata, flavor="seurat", n_top_genes=2000)

# Perform Principal Component Analysis (PCA) on the AnnData object
sc.pp.pca(count_area_filtered_adata)

# Build a neighborhood graph based on PCA components
sc.pp.neighbors(count_area_filtered_adata)

# Perform Leiden clustering on the neighborhood graph and store the results in 'clusters' column

# Adjust the resolution parameter as needed for different samples
# datasetによって調整必要
sc.tl.leiden(count_area_filtered_adata, resolution=0.35, key_added="clusters")

```
⑤ クラスタープロット

```
# Plot and save the clustering results
plot_clusters_and_save_image(title="Region of interest 1", gdf=gdf, img=img, adata=count_area_filtered_adata, bbox=(12844,7700,13760,8664), color_by_obs='clusters', output_name=dir_base+"image_clustering.ROI1.tiff")
```
<img src="https://aw-lab.notepm.jp/private/c2584d70-42a3-11ef-8a38-064017533d40.png" title="clustering results plot" width="640">



⑥ 特定の遺伝子発現プロット

turtorialでは、例としてLyz1、Muc2、Fabp2、Jchainの遺伝子を見ている。
例えば、Lyz1のプロットのコードは以下であり、 gene='Lyz1'の''内の名前を変更することで特定の遺伝子発現細胞を同定できる。

```
# Plot Lyz1 gene expression
plot_gene_and_save_image(title="Region of interest 1", gdf=gdf, gene='Lyz1', img=img, adata=count_area_filtered_adata, bbox=(12844,7700,13760,8664),output_name=dir_base+"image_Lyz1.ROI1.tiff")
```
<img src="https://aw-lab.notepm.jp/private/e76e1572-42a3-11ef-b37f-061da1ef3444.png" title="gene expression plot_Lyz1" width="640">






