Marvin's Appendix
=================

- Create trajectory (series of arbitrary points) in euclidian space
- Transform trajectory to actuator space
- Interpolate transformed trajectory with splines (you need the first two derivatives as well)
- Reparametrize splines with constant arc lengths
- Calculate maximal speed (due to centrifugal forces) and acceleration in the direction of the trajectory in function of the position along the parametrization
- Deform the function of the maximal speed to stay within the acceleration limits along the trajectory
- Resample the result in the time domain
- You now have a time associated to positions along the parametrization, as well as the speed and acceleration
- Evaluating the splines (and their derivatives) at these positions yeilds points in the actuator space and the direction of the velocity and accleration
- Using this, calculate the feed-forward torque
