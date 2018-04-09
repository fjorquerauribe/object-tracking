# Object Tracking

Object Tracking project

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- OpenCV 3.3
- MXNet (https://github.com/apache/incubator-mxnet)
- Scikit-learn

## Installation

Install rcnn example (https://github.com/apache/incubator-mxnet/tree/master/example/rcnn)

## Running the test

Download the project:
```
git clone https://github.com/fjorquerauribe/object-tracking.git
cd object-tracking
```
Download the MOT Challenge datasets https://motchallenge.net/ or VOT Challenge datasets http://www.votchallenge.net/

Create a symbolic link to the datasets folder:
```
cd scripts
ln -s path/to/datasets/ data
```

Example: run Bernoulli Particle filter over car sequence:
```
./start_bpf.sh vot2014 car 100
```

## Disclaimer

This repository used code from (MXNet)(https://github.com/apache/incubator-mxnet).

## Author

* **Felipe Jorquera Uribe** - [fjorquerauribe](https://github.com/fjorquerauribe)

See also the list of [contributors](https://github.com/fjorquerauribe/multitarget-tracking/graphs/contributors) who participated in this project.

## License

This project is licensed under the Apache 2 license - see the [LICENSE.txt](LICENSE.txt) file for details.