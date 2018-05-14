# _COBALT Tractography_
Package that performs tractography on 3D tiff stack volumes

## System Requirements
Python 2.7
docker

### Pull and run docker
A docker image with python 2.7 needs to be pulled and run. The following docker is optimized for multi-core Intel processors:
```docker pull neurodata/ndreg```
Then run:
```docker run -p 8888:8888 neurodata/ndreg```
Then execute /bin/bash inside the docker:
```
docker exec -it <docker name> /bin/bash/
```

### Software Dependencies
The following python packages are needed. They will be automatically downloaded and installed once you pip install the package:

scikit_image <br/>
scipy <br/>
numpy <br/>
requests <br/>
intern <br/>
tifffile<br/>
matplotlib <br/>
scikit_learn <br/>
scikit-fmm <br/>

## Installation
First clone this repository:
```git clone https://github.com/neurodata-cobalt/ndtractography.git```

Then run ``` pip install .``` inside ndtractography directory to install the package and its requirements

## Use
You can now use the functions by importing the following in your python script:
```
from cobalt_tractography.bossHandler import *
from cobalt_tractography.tractography import *
```


