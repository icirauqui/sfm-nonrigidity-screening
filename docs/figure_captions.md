# Figure captions and alternative text — 25 October 2026

The typeset article captions are authoritative. The complete figures retain their original scientific support.

## Figure 1 — core_tubes_views

Controlled scene families and retained image views. Panels (a--d) show f01 straight, f02 right bend, f03 upward bend and f04 two bends. Each pairs the source wall geometry (left) with the static moving-camera RGB view at frame 52 (right). Meshes share one orthographic projection and one generating-coordinate scale normalized by the f01 reference radius $R_{\mathrm{f01}}$. Each projected mesh is translated to the center of identical display bounds; its shape, aspect ratio and scale are preserved. The $4R_{\mathrm{f01}}$ scale bar applies to all four meshes; the other families need not have that radius. RGB uses a common square-root display transform. Source surfaces and images illustrate acquisition rather than supply detector inputs. Codex assisted the visualization code.

**Alternative text:** Four panels in a two-by-two grid pair larger source meshes with retained static RGB views of f01–f04. Meshes are translated to a common center at identical scale and projection; one shared radius scale bar applies to all.

## Figure 2 — bundles_geometry

Actual Bundles evidence for f01/FE-derived motion. (a) All 62,436 global points, including outliers, and the 90 estimated camera centers; the dark trajectory is global and orange highlights the selected local window. (b) Detail with fixed 1st--99th percentile coordinate bounds, marked in (a). (c,d) All 10,964 matched global/local points for window 4, images 40--54, with separately estimated or inherited cameras. Each local model is aligned by the unchanged camera-derived similarity; purple shows the aligned local camera path and orientation glyphs. The paired panels share bounds; no surface or truth alignment is fitted. Coordinates use a common display rotation and estimated global extent $L_g$. Glyph depth is fixed for display and does not represent calibrated fields of view. Point colors in (a,b) come from the retained export; teal/orange in (c,d) distinguish paired models. Codex assisted the visualization code.

**Alternative text:** Complete estimated f01 FE global point cloud and camera path, a labeled percentile detail, and two aligned local point comparisons for images 40–54. All points in each stated support remain.

## Figure 3 — global_modal_explanation

Global modal reprojection. The flow diagram summarizes the single-model computation. (a) All retained tracked observations in f01/FE-derived image 44, colored by their global reprojection-residual norm; 60 fixed ordinal residual arrows are enlarged 20 times for visibility. Values above 1.5 pixels share the color-scale endpoint and remain in the data. (b,c) Schematic affine and quadratic basis-field examples on the source f01 tube, with arbitrary amplitude before pixel projection and nuisance removal. These basis illustrations are not recovered motion or data used by inference. The actual operator uses estimated global coordinates, 24 affine temporal nuisance columns and 36 quadratic alternative columns. Codex assisted the visualization code.

**Alternative text:** A flowchart connects image residuals, nuisance removal and modal energy. One actual residual image-plane plot is separated from two schematic affine/quadratic tube-basis examples.

## Figure 4 — rdsc_spatial_fields

Spatial intermediate fields of RDSC. Rows show window 4 in f01 and f03 FE-derived acquisitions with separately estimated local cameras; every matched point is retained. Color encodes the norm of (a,d) aligned discrepancy $d$, (b,e) its robust affine residual $e$, or (c,f) the nine-entry neighborhood mean $c$, divided by the original-export matched extent $L_w$. Within each row, display coordinates and bounds are shared across stages; the rows are separately centered. The common logarithmic scale saturates outside $[10^{-6},10^{-1}]$ for display only. Vector directions are not shown and scores are unchanged. These are reconstruction discrepancies rather than measured material-displacement vectors. Codex assisted the visualization code.

**Alternative text:** Two complete matched-point examples show norms of aligned discrepancies, robust affine residuals and neighborhood means on a common logarithmic scale. These are reconstruction differences, not measured tissue motion.

## Figure 5 — basc_provenance_routing

Camera provenance and BASC-C composition. (a) In the f01/FE-derived window 4 example, estimated local camera-center differences are shown after camera-derived alignment and normalization by $L_g$. The inherited-camera curve is zero to roundoff because the camera inputs are shared; the separate-camera curve is a reconstruction difference, not pose error against truth. (b) The fixed provenance branches use calibrated sequence decisions. A decisive preferred answer is retained; RDSC is used only when that answer is unavailable. Unknown or mixed provenance abstains. The global modal readout is a separate strategy, not another expert in this cascade. Codex assisted the visualization code.

**Alternative text:** An example camera-center discrepancy curve accompanies the fixed separate/inherited-camera routing branches. Decisive preferred answers are preserved; unavailable answers use RDSC fallback.

## Figure 6 — controlled_motion_views

Source motion conditions, shown for f01 at frame 52. Upper panels show the stored static, FE-derived and prescribed non-FE surfaces at true deformation scale, colored by displacement magnitude divided by the f01 reference radius $R$; the gray wireframe is the rest wall. Lower panels show the retained moving-camera RGB views at the same camera pose, using the same square-root display transform. The two positive conditions have different prescribed fields; similar image appearance does not imply static geometry. These evaluator/source views explain acquisition labels and are not used to construct a screening score. FE-derived kinematics do not establish tissue-mechanical realism. Codex assisted the visualization code.

**Alternative text:** Actual static, FE-derived and prescribed non-FE f01 surfaces at frame 52, with matched RGB views. Surface colors encode stored displacement norm at true geometric scale.

## Figure 7 — core_global_readouts

Complete global-readout comparison. Scores are divided by each method's family-excluded static cutoff; values above one screen positive. (a) Modal energy and its studentized readout; (b) residual-magnitude summaries, shown on an expanded scale near the cutoff. The panels intentionally use different vertical scales. FE and prescr. identify the two prescribed-motion conditions. Every acquisition is shown. Energy and $F$ make identical decisions; the single static false positive follows the maximum-calibration design. Codex assisted the plotting code.

**Alternative text:** All twelve core acquisitions show two modal and three magnitude readouts relative to their own fixed static cutoffs. Two panels use explicitly different vertical scales.

## Figure 8 — core_complete_decisions

Complete core decision matrix. The same twelve acquisitions appear under separate and inherited local-camera estimation. LOG loc./ori./union denote the LOG-lite/ZD location/orientation/union adaptations. A $+$ or $-$ gives a positive or negative screen; blue denotes a correct answer and orange an incorrect answer relative to the supplied acquisition label. The left-hand labels give that class as well. This is descriptive paired accounting; an either-correct oracle is not used for inference. Codex assisted the plotting code.

**Alternative text:** Signs and redundant colors display all twelve acquisition decisions under both local-camera regimes, including all incorrect answers. The regimes repeat the same acquisitions.

## Figure 9 — appearance_complete_transfer

Complete positive-only appearance/pipeline transfer. Cells show score divided by the fixed static cutoff; color shows its base-10 logarithm. Values are rounded to two decimals, while appended $+$ or $-$ signs use the unrounded decision. Gray A denotes the retained point-information-rank abstention. All twenty unique exports are shown; the two pipelines and five appearances reuse two core families and contain no matched appearance negatives. Codex assisted the plotting code.

**Alternative text:** All twenty positive appearance/pipeline exports show five normalized scores. Explicit unrounded decision signs accompany rounded ratios; one whole-model abstention is retained.

## Graphical abstract

Retained estimated SfM evidence feeds three alternative screening strategies: global modal residuals, RDSC spatial coherence and BASC-C expert composition. Each returns a screen or abstention; conditional trade-offs are stated. Retained-model detail uses fixed percentile bounds; code assistance is disclosed in the figure.
