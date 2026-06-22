# 01 Download Subset Notebooks

This folder contains the 20 successful Kaggle Notebook 01 downloads used by the Strong Official-Mini GR00T pipeline.

Naming convention:

`01_download_XX_<TaskName>_ego<N>.ipynb`

- `ego<N>` means the notebook downloaded `observation.images.ego_view` video chunks only, not full multi-camera video.
- Most subsets use chunks `0,1,2`; `PlateToBowl` and `CuttingboardToTieredBasket` use chunks `0,1,2,3,4`.
- Metadata and action/state parquet data are included for each subset.

Removed failed duplicate:

- `notebookbf44cf0201.ipynb`: duplicate `gr1_arms_waist.CuttingboardToTieredBasket`, report status was `error`, `downloaded_ok=false`.

## Summary

- Successful notebooks: 20
- Download status: all remaining notebooks have `status=downloaded` and `downloaded_ok=true`
- Reported total size: about 119.36 GB
- Reported video files: 61,999
- Reported parquet files: 192,226

## Notebook Map

| # | Subset | Chunks | Local file |
|---:|---|---:|---|
| 01 | `gr1_arms_waist.PlaceMilkToMicrowave` | 3 | `01_download_01_PlaceMilkToMicrowave_ego3.ipynb` |
| 02 | `gr1_arms_waist.PotatoToMicrowave` | 3 | `01_download_02_PotatoToMicrowave_ego3.ipynb` |
| 03 | `gr1_arms_waist.PlateToBowl` | 5 | `01_download_03_PlateToBowl_ego5.ipynb` |
| 04 | `gr1_arms_waist.PlateToPlate` | 3 | `01_download_04_PlateToPlate_ego3.ipynb` |
| 05 | `gr1_arms_waist.TrayToPlate` | 3 | `01_download_05_TrayToPlate_ego3.ipynb` |
| 06 | `gr1_arms_waist.WineToCabinet` | 3 | `01_download_06_WineToCabinet_ego3.ipynb` |
| 07 | `gr1_arms_waist.CuttingboardToBasket` | 3 | `01_download_07_CuttingboardToBasket_ego3.ipynb` |
| 08 | `gr1_arms_waist.CuttingboardToPan` | 3 | `01_download_08_CuttingboardToPan_ego3.ipynb` |
| 09 | `gr1_arms_waist.PlaceBottleToCabinet` | 3 | `01_download_09_PlaceBottleToCabinet_ego3.ipynb` |
| 10 | `gr1_arms_waist.PlacematToBowl` | 3 | `01_download_10_PlacematToBowl_ego3.ipynb` |
| 11 | `gr1_arms_waist.TrayToPot` | 3 | `01_download_11_TrayToPot_ego3.ipynb` |
| 12 | `gr1_arms_waist.TrayToTieredShelf` | 3 | `01_download_12_TrayToTieredShelf_ego3.ipynb` |
| 13 | `gr1_arms_only.CanSort` | 3 | `01_download_13_CanSort_ego3.ipynb` |
| 14 | `gr1_arms_waist.CanToDrawer` | 3 | `01_download_14_CanToDrawer_ego3.ipynb` |
| 15 | `gr1_arms_waist.CupToDrawer` | 3 | `01_download_15_CupToDrawer_ego3.ipynb` |
| 16 | `gr1_arms_waist.CuttingboardToPot` | 3 | `01_download_16_CuttingboardToPot_ego3.ipynb` |
| 17 | `gr1_arms_waist.CuttingboardToTieredBasket` | 5 | `01_download_17_CuttingboardToTieredBasket_ego5.ipynb` |
| 18 | `gr1_arms_waist.PlacematToPlate` | 3 | `01_download_18_PlacematToPlate_ego3.ipynb` |
| 19 | `gr1_arms_waist.PlateToPan` | 3 | `01_download_19_PlateToPan_ego3.ipynb` |
| 20 | `gr1_arms_waist.TrayToCardboardBox` | 3 | `01_download_20_TrayToCardboardBox_ego3.ipynb` |

See `download_subset_manifest.json` for the machine-readable manifest.
