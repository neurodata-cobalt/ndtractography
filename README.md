# _COBALT's ndTractography_
Package for tractography on 3D neuronal images

## System Requirements
Python 2.7 <br/>
docker <br/>
Currently tested on ubuntu 16

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

### Pull and run docker
A docker image with python 2.7 needs to be pulled and run. The following docker is optimized for multi-core Intel processors:

```docker pull neurodata/ndreg```

Then run:

```docker run -p 8888:8888 neurodata/ndreg```

Then execute /bin/bash inside the docker:

```
docker exec -it <docker name> /bin/bash/
```

### Clone and pip install

First clone this repository:
```git clone https://github.com/neurodata-cobalt/ndtractography.git```

Then run ``` pip install .``` inside ndtractography directory to install the package and its requirements

### Adding BOSS API token

Save your token at `~/.intern/intern.cfg` with the following format:

```
[Default]
protocol = https
host = api.boss.neurodata.io
token = <your BOSS API token>
```

## Use
You can now use the functions by importing the following in your python script:
```
from cobalt_tractography.bossHandler import *
from cobalt_tractography.tractography import *
```

## Resources
* [ndtractography algorithm pseudocode and flow diagram](https://github.com/neurodata-cobalt/ndtractography/wiki)
