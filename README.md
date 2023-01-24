# SWOT error calibration Gulf Stream 2022c
A challenge on the SWOT satellite error calibration organised by Datlas, IGE, IMT Altlantique and CLS.


<img src="figures/DC_SWOT_error_calibratio _GS-banner.png" />

## Plan 

[**1. Context & Motivation**](#contextmotivation)

[**2. The experiment**](#theexperiment)

&ensp; [2.1 Experimental setup](#sec21)
    
&ensp; [2.2 Baseline](#sec23)
    
&ensp; [2.3 Evaluations](#sec23)
    
[**3. To get started**](#togetstarted) 

&ensp; [3.1 Installation](#sec31)
    
&ensp; [3.2 Download the data](#sec32)
   
[**4.To compare**](#tocompare)

&ensp; [4.1 Method descriptions](#sec41)
    
&ensp; [4.2 Evaluation on SWOT swath: Leaderboard](#sec42)
    
&ensp; [4.3 Evaluation on Nadirs tracks: Leaderboard](#sec43)
     
[**Acknowledgement**](#acknowledgement)

[**References**](#references)


<a name="contextmotivation"></a>
## 1. Context & Motivation

The two-dimensional sea level SWOT products are very much expected to be a game changer in many oceanographic applications which will make them an unprecedented L3 product to be distributed. The row SWOT data will however be contaminated by instrumental and geophysical errors ([Gaultier et al., 2016](https://journals.ametsoc.org/view/journals/atot/33/1/jtech-d-15-0160_1.xml?tab_body=fulltext-display); [Peral and Esteban-Fernandez, 2018](https://ieeexplore.ieee.org/document/8517385/authors#authors)). In order to be able to observe front, mesoscale and sub-mesoscale features, the SWOT data will require specific processing. Also, these errors are expected to strongly pollute the first and second derivatives of the SSH data which are used for the computation of geostrophic currents and vorticity. Hence, being able to remove the SWOT errors will be of significant importance to recover information on 2D surface currents and vertical mixing.  

The SWOT errors are expected to generate noises that are both correlated on the swath and spatially uncorrelated. Several past efforts have already investigated methods to remove or reduce the correlated noises from the SWOT data using prior knowledge on the ocean state (e.g. [Metref et al., 2019](https://www.mdpi.com/2072-4292/11/11/1336), see Figure 2.4.A), calibration from independent Nadir altimeter data (e.g. Febvre et al., 2021, see Figure 2.4.B) or cross-calibration from SWOT data themselves (on-going CNES-DUACS studies). And other efforts focused on reducing the uncorrelated data (Gomez-Navarro et al., [2018](https://www.mdpi.com/2072-4292/10/4/599), [2020](https://www.mdpi.com/2072-4292/12/4/734); Febvre et al., 2021). Yet, so far, no rigorous intercomparison between the recently developed methods has been undertaken and it seems difficult, to this day, to outline the benefits and limitations of favoring one error reduction method from another.

It is important to mention that the SWOT science requirement for uncorrelated noise measurement error specifies that KaRin must resolve SSH on wavelength scales up to 15 km based on the 68th percentile of the global wavenumber distribution.

**The goal of this SWOT Calibration data challenge is to provide a platform to investigate the most appropriate calibration procedures to reduce the <span style="color:red">spatially correlated</span> errors from the SWOT data.** 

You can find another data challenge proposed by Datlas, CLS and IMT-Atlantique if you are interested in filtering only the instrumental (KaRIn) SWOT noise: [here](github.com/ocean-data-challenges/2022a_SWOT_karin_error_filtering).

In practice, the SWOT error calibration data challenge is in the form of an Observing System Simulation Experiment (OSSE) in the Gulf Stream region considering a realistic ocean model simulation (NATL60) as the true ocean state. The SWOT simulator ([Gaultier et al., 2016](https://journals.ametsoc.org/view/journals/atot/33/1/jtech-d-15-0160_1.xml?tab_body=fulltext-display)) was then used to create realistic SWOT data with and without correlated errors. Then, various calibration procedures can be tested and compared to the true ocean state.

This data challenge is partly funded by the CNRS, by the CNES and, as part of the **Sea Level Innovations and Collaborative Intercomparisons for the Next-Generation products (SLICING)** project, by the Copernicus Marine Service Evolution (21036-COP-INNO SCI).  

<a name="theexperiment"></a> 
## 2. The experiment

<a name="sec21"></a>
### 2.1 Experimental setup

The data challenge is in the form of an Observing System Simulation Experiment (OSSE) considering a realistic ocean model simulation, the NEMO high resolution North Atlantic simulation **NATL60**, as the true ocean state in a 10°X10° degree box in the Gulf Stream region. The SWOT simulator ([Gaultier et al., 2016](https://journals.ametsoc.org/view/journals/atot/33/1/jtech-d-15-0160_1.xml?tab_body=fulltext-display)) was then used to create realistic SWOT data with and without instrumental noise. 

The experiment is performed over one SWOT orbital cycle (cycle 13) which contains 270 passes. All other cycles are available to tune or train the filters.

The SWOT data to calibrate (the inputs: **ssh_err**) and their equivalent error-free SWOT data for evaluation (the targets: **ssh_true**) are hosted and available for download on the MEOM opendap server: see **Download the data** section below.
In no way the targets that are available during the evaluation period should be used in the calibration process.   


<img src="figures/DC_SWOT_error_calibration_GS-illustration.png" /> 
<center><em>Fig.: Example of one target of the data challenge: SWOT data without errors (left), one input of the data challenge: SWOT data with errors (center) and the difference between the two (right). Here, SWOT is "flying" over the NATL60 model simulation in the Gulf Stream region.
</em></center>

<a name="sec22"></a>
### 2.2 Baseline 

The baseline method is a very simple and straightforward method that removes the averaged on the pass of a linear trend across track. This is supposed to remove all the signal that has the same structure as the roll error (if the roll error was constant over the region). This method is flowed since (1) there are other errors, (2) the roll error is not constant over the region and (3) the method also removes signals that happens to have a linear structure across track but is actual SSH. 
The baseline is performed in the notebook [demo_generate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/demo_generate_baseline.ipynb).


<a name="sec23"></a>
### 2.3 Evaluations

There are two ways of evaluating the calibration performance. Since the experiment is an OSSE, we do have the true SSH on the SWOT swath. Hence the first possible evaluation is directly on the SWOT swath where the calibrated SWOT data are compared to the true SWOT data. The second evaluation is performed using OSE-type metrics, i.e. as if the true data on the swath wasn't available. In this case, the calibrated SWOT data are interpolated onto some Nadir tracks (envisat, geosat2, jason1 and topex-poseidon) and then compared to the true SWOT data (also interpolated on the Nadir tracks).  


#### Evaluation on SWOT swath

The first evaluation of the calibration methods is based on the comparison of the SWOT calibrated swaths with the error-free SWOT swaths. It includes two scores, one based on the Root-Mean-Square Error (RMSE), the other a noise-to-signal ratio based on an isotropic 2D Fourier wavenumber spectra. The evaluation notebook [demo_evaluate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_evaluate_on_swot/demo_evaluate_onswot_baseline.ipynb) implements the computation of these two scores for the baseline as they appear in the first leaderboard. 

After running the evaluation notebook for a method, the files stat_'METHOD'.nc (RMSE data), psd_'METHOD'.nc (PSD and SNR data) and ldb_'METHOD'.nc (Leaderboard data) are saved in the chosen foler. 

A comparison notebook, [compare_evaluations_onswot](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_evaluate_on_swot/compare_evaluations_onswot.ipynb), is available to plot and compare the statistics and the spectral diagnostics of multiple methods in the same figures. Also, the leaderboard for each method can be printed. 


#### Evaluation on Nadirs tracks

The second evaluation of the calibration methods is based on the comparison of the true SWOT data interpolated on Nadir tracks. It also includes two scores, one based on the Root-Mean-Square Error (RMSE), the other a noise-to-signal ratio based on a one-dimensional (along Nadir tracks crossing the swath) Fourier wavenumber spectra. The evaluation notebook [demo_evaluate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_evaluate_on_nadirs/demo_evaluate_onnadirs_baseline.ipynb) implements the computation of these two scores for the baseline as they appear in the second leaderboard. 


<a name="togetstarted"></a>
## 3. To get started

<a name="sec31"></a>
### 3.1 Installation
:computer: _**How to get started ?**_

Clone the data challenge repo: 
```
git clone https://github.com/ocean-data-challenges/2022c_SWOT_error_calibration_GS.git
```
create the data challenge conda environment, named env-dc-swot-filtering, by running the following command:
```
conda env create --file=environment.yml 
```
and activate it with:

```
conda activate env-dc-swot-calibration-gs
```
then add it to the available kernels for jupyter to see: 
```
ipython kernel install --name "env-dc-swot-calibration-gs" --user
```

You're now good to go !

[Check out the quickstart](quickstart.ipynb)


<a name="sec32"></a>
### 3.2 Download the data

The data are hosted on the opendap: [ocean-data-challenges/2022c_SWOT_error_calibration_GS/](https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/catalog/meomopendap/extract/ocean-data-challenges/2022c_SWOT_error_calibration_GS/catalog.html). 
 

The data needed for the DC are presented with the following directory structure:

```
.
|-- 2022c_SWOT_error_calibration_GS
|   |-- dc_SWOTcalibGS_inputs.tar.gz
|   |-- dc_SWOTcalibGS_nadirs.tar.gz
|   |-- dc_SWOTcalibGS_maps.tar.gz

```


To start out download the datasets from the data server.

#### SWOT data (inputs and targets)
```shell
!wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022c_SWOT_error_calibration_GS/dc_SWOTcalibGS_inputs.tar.gz

```
and then uncompress the files using `tar -xvf <file>.tar.gz`. You may also use `ftp`, `rsync` or `curl`to donwload the data.
**The inputs are stored in the variable *ssh_err* and the targets are stored in the variable *ssh_true*.**


#### Other potentially useful data (nadirs and maps)

##### 5 nadirs data (for calibration if needed and evaluation on nadir tracks)
```shell
!wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022c_SWOT_error_calibration_GS/dc_SWOTcalibGS_nadirs.tar.gz

```

##### Maps data: OI maps made from the 5 nadirs data (for calibration if needed)
```shell
!wget https://ige-meom-opendap.univ-grenoble-alpes.fr/thredds/fileServer/meomopendap/extract/ocean-data-challenges/2022c_SWOT_error_calibration_GS/dc_SWOTcalibGS_maps.tar.gz

```
and then uncompress the files using `tar -xvf <file>.tar.gz`. You may also use `ftp`, `rsync` or `curl`to donwload the data. 

 



<a name="tocompare"></a>
## 4. To compare 


<a name="sec41"></a>
### 4.1 Method descriptions

#### Baseline

See the **Baseline** section above. 

#### CER method

The CER method is based on [Metref et al. (2019)](https://www.mdpi.com/2072-4292/11/11/1336). The CER method algorithm is included in src/aux.py and launched in src/swot.py with the function `apply_CERmethod_calib`.

Here, the ensemble used for the assimilation is a 1/20° SSH fields created using optimal interpolations with Nadir altimeter data (both the maps and the Nadir data are available on the opendap ; check **Download the data** section). 
From these maps, a 10-member ensemble is generated by selecting fields at dates close to the SWOT observation date. 
The ensemble is projected onto the SWOT swath and both the SWOT data and the projected ensemble are detrended (i.e., the signal with a geometrical structure similar to the expected error structure is removed). 
Using the projected ensemble, the detrended ensemble and the detrended SWOT data, an assimilation analysis is performed (using an Ensemble Transform Kalman Filter). 
The ensemble average of this analysis constitutes the calibrated SWOT data. 



#### Projection method

The projection method is very similar to the CER method. The projection method interpolates the SSH map created with the OI method and 5 nadirs (see CER method) onto the SWOT track. The difference between the SWOT with error data and the interpolated SSH map is then "projected", i.e., using a min square method the signal in this difference that has the same geometry as the expected SWOT errors (linear across track for the roll error, quadratic for the baseline dilation error ...). The residual is then added back onto the interpolated OI SSH map (containing mostly larger scales). 
A low-pass and high-pass filter is also performed so to leave untouch the mesoscale that is already accurate in the OI SSH map and that can be wrongly projected onto the error geometry. 

#### CalCNN

The CalCNN method is based on Febvre et al. (in prep.). The algorithms are not yet shared by the authors but will be soon. 

The CalCNN method uses a trainable architecture that takes as inputs the uncalibrated observations and the nadir based L4 product interpolated on the SWOT swath. 
The architecture is trained in a supervised manner on the reconstruction of the SWOT SSH. 
The L4 product and KaRIN observations are first each decomposed into Ns scale components which are concatenated together as channels. 
This produces an input tensor of shape (2Ns, Nal, Nac) where Nal and Nac are respectively the along track and across track sizes of the input swath section. 
An input convolution layer is used to change the number of channels. The data is then processed by a series a residual convolutional blocks composed of a convolution layer, a ReLU non-linearity, a skip connection and a mixing layer. 
Then a last convolution layer is used to compute a point wise correction of the L4 product from the different channels. 
Finally this correction is added to the input interpolated field to produce the calibrated SWOT data.


<a name="sec42"></a> 
### 4.2 Evaluation on SWOT swath: Leaderboard

| Method   | Field                          |   µ(RMSE) |    λ(SNR1) [km] | Reference                  |
|:---------|--------:|-----------------:|-------------------------------:|:---------------------------|
| **No Calib**   | Sea Surface Height $[m]$|        3.395 |         116.3  |  --- |  
| ---   | Geos. current $[m.s^{-1}]$|        9.437 |         nan  |  --- |  
| ---   | Relative vorticity $[]$|        0.760 |         37.9  |  --- | 
| **Baseline**   | Sea Surface Height $[m]$|        0.145 |         83.9  |  [demo_generate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/demo_generate_baseline.ipynb) |  
| ---   | Geos. current $[m.s^{-1}]$|        0.288 |         84.9  |  [demo_generate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/demo_generate_baseline.ipynb) |  
| ---   | Relative vorticity $[]$|        0.529 |         36.5  |  [demo_generate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/demo_generate_baseline.ipynb) | 
| **CER-method**   | Sea Surface Height $[m]$|        0.074 |        **4.2**  |  [generate_CERmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_CERmethod_calibration.ipynb) |  
| ---   | Geos. current $[m.s^{-1}]$|        0.219 |         78.4  |  [generate_CERmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_CERmethod_calibration.ipynb) |  
| ---   | Relative vorticity $[]$|        0.261 |         103.4  |  [generate_CERmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_CERmethod_calibration.ipynb) | 
| **Projmethod**   | Sea Surface Height $[m]$|        0.037 |        **4.2**  |  [generate_Projmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_Projmethod_calibration.ipynb) |  
| ---   | Geos. current $[m.s^{-1}]$|        0.115 |      40.2 |  [generate_Projmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_Projmethod_calibration.ipynb) |  
| ---   | Relative vorticity $[]$|        0.271 |        **30.6**  |  [generate_Projmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_Projmethod_calibration.ipynb) | 
| **CalCNN**   | Sea Surface Height $[m]$|        **0.014** |         **4.2**  |  Offline generation |  
| ---   | Geos. current $[m.s^{-1}]$|        **0.076** |         **30.6**  | Offline generation    |  
| ---   | Relative vorticity $[]$|        **0.212** |         37.9  | Offline generation | 

  

with:

 `µ(RMSE)`: averaged root-mean square error over the tracks crossing the swath, 
 
 `λ(SNR1)`: spatial wavelength where SNR=0.5.
 

<a name="sec43"></a>
### 4.3 Evaluation on Nadirs tracks: Leaderboard

| Method   | Field                          |   µ(RMSE) |    λ(SNR1) [km] | Reference                  |
|:---------|--------:|-----------------:|-------------------------------:|:---------------------------| 
| **Ref (True SWOT)**  | Sea Surface Height $[m]$|     0.014  |    15.561  |  --- |  
| **No Calib**   | Sea Surface Height $[m]$|    4.041   |    NaN  |  --- |  
| **Baseline**   | Sea Surface Height $[m]$|        0.218 |   30.270 |  [demo_generate_baseline](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/demo_generate_baseline.ipynb) |  
| **CER-method**   | Sea Surface Height $[m]$|        0.074 |   **23.800**   |  [generate_CERmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_CERmethod_calibration.ipynb) |  
| **Projmethod**   | Sea Surface Height $[m]$|        **0.037** |  25.647  |  [generate_Projmethod_calibration](https://github.com/SammyMetref/2022c_SWOT_error_calibration_GS/blob/main/notebooks_generate_calibrations/generate_Projmethod_calibration.ipynb) |  
| **CalCNN**   | Sea Surface Height $[m]$|        --- |        ---  |  Offline generation |  

 

with:

 `µ(RMSE)`: averaged root-mean square error over the swaths 
 
 `λ(SNR1)`: spatial wavelength where SNR=0.5.

 
<a name="acknowledgement"></a>
## Acknowledgement

This data challenge was funded by the CNRS, by the CNES and by the Service Evolution CMEMS project: SLICING, in collaboration with Datlas, IGE, CLS, IMT-Atlantique.

The structure of this data challenge was to a large extent inspired by the [`ocean-data-challenges`](https://github.com/ocean-data-challenges) created for the BOOST-SWOT ANR project.
 



<a name="references"></a>
## References
 
 
Gaultier, L., Ubelmann, C., & Fu, L. L. (2016). The challenge of using future SWOT data for oceanic field reconstruction. Journal of Atmospheric and Oceanic Technology, 33(1), 119-126.

Gómez-Navarro, L., Fablet, R., Mason, E., Pascual, A., Mourre, B., Cosme, E., & Le Sommer, J. (2018). SWOT spatial scales in the western Mediterranean sea derived from pseudo-observations and an Ad Hoc filtering. Remote Sensing, 10(4), 599.

Gomez-Navarro, L., Cosme, E., Le Sommer, J., Papadakis, N., & Pascual, A. (2020). Development of an image de-noising method in preparation for the Surface Water and Ocean Topography satellite mission. Remote Sensing, 12(4).

Metref, S., Cosme, E., Le Sommer, J., Poel, N., Brankart, J. M., Verron, J., & Gómez Navarro, L. (2019). Reduction of spatially structured errors in wide-swath altimetric satellite data using data assimilation. Remote Sensing, 11(11), 1336.

Peral, E., & Esteban-Fernandez, D. (2018, July). SWOT mission performance and error budget. In IGARSS 2018-2018 IEEE International Geoscience and Remote Sensing Symposium (pp. 8625-8628). IEEE.