# SUMO Dynamic Origin-Destination Matrix Estimation v0.1.0

A codebase containing different tools for estimating origin-destination matrices using SUMO.

## Overview

Currently the main implementation resides in the `/fileserver` directory:

- The `/fileserver/main.py` contains code that can be executed to: 
  - Generate a test traffic network
  - Generate synthetic scenario data using a generated test traffic network
  - Run all or a specific estimation algorithm of choice given given synthetic scenario data as input   
- The `/fileserver/static` directory contains experimental data and final algorithm outputs
- The `/fileserver/images` directory contains different plots generated with python scripts `/fileserver/plot1.py` and `/fileserver/plot2.py`

Assuming SUMO (>=1.10.0) and all Python dependencies have been intalled, the `/fileserver/images/main.py` can be executed:
```bash
# Run database migrations
python manage.py makemigrations && python manage migrate

# Run estimation script
python main.py
```
## Note

To be able to run the estimation procedure the `duaIteratev3.py` script contained in this current
directory will have to be copied to the same directory containing the original `duaIterate.py` installed 
together with SUMO. The `duaIterate.py` script usually resides in directory `/usr/share/sumo/tools/assign/`.

Running the following command should do it:

```bash
cp duaIteratev3.py /usr/share/sumo/tools/assign/.
```

## Implementation Notes

To be consistent acorss methods and the data used by them data is handled internally:

- Vehicle link counts are derived from vehicle exist times recorded in vehicle [route files](https://sumo.dlr.de/docs/Simulation/Output/VehRoutes.html) 
- Vehicle link counts are aggregate meaning that counts are aggregated across lanes if multiple lanes exist on a link

