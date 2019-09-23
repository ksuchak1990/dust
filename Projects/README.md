# DUST Projects
A master folder for dust-related sub-projects.  [DUST](https://dust.leeds.ac.uk/) ([repo](https://github.com/Urban-Analytics/dust/), [gist](https://urban-analytics.github.io/dust/), [OneDrive](https://leeds365-my.sharepoint.com/personal/geonsm_leeds_ac_uk/_layouts/15/onedrive.aspx), [Slack](https://lida-uol.slack.com/), [Trello](https://trello.com/b/2WMzu1tt/)),  Dynamic Urban Simulation Techniques is a research group exploring the uses of Data Assimilation (DA) for Agent-Based Modelling (ABM).


## Models
`model.py` - A minimalist framework for building ABMs suitable for the DA side of this projects.  Based on [mesa](https://github.com/projectmesa/mesa).
- Andrew West, gyawe
- Keiran Suchak, k.suchak

##### BusSim
Bus route simulation.
- Minh Kieu, m.l.kieu

##### StationSim
Pedestrian movement in a crowded corridor.  
[Originally developed](https://github.com/nickmalleson/keanu-post-hackathon/tree/stationSim/keanu-examples/stationSim/src/main/java/StationSim) in Java and Keanu, this is a python edition.
- Andrew West, gyawe


## Filters
`filter.py` - A framework for ensuring models and filters are compatible.
- Keiran Suchak, k.suchak
- Andrew West, gyawe

##### Particle Filter
- Kevin Minors, k.minors

##### Ensemble Kalman Filter
- Keiran Suchak, k.suchak

##### Unscented Kalman Filter
- Robert Clay, r.clay


## Experiments
Master published experiments should be done in a [notebook](https://mybinder.org/v2/gh/Urban-Analytics/dust/master?filepath=Projects%2FExperiments).
- Undesignated


## Probabilistic
A folder dedicated to the probabilistic programming side of this project.
- Nick Malleson, n.s.malleson
- Benjamin Wilson, b.wilson1


## Resources
- [email]@leeds.ac.uk
- [R. Labbe](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) (2015) _Kalman and Bayesian Filters in Python_
