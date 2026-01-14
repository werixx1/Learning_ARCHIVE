<h3 align="center">Hardware in the loop</h3>

  <p align="center">
    Intuitive understanding and application of HIL
    <br>
    <a href="">Resources</a>
    ·
    <a href="">Main</a>
  </p>
</p>


## Table of contents

- [Explanation](#explanation)


## Explanation
> Key words: real-time testing, simulation, automotive

HIL is testing method for embedded systems, connecting a **real controller** (like a car's ECU) (I/O pins) to a fast **real-time simulator** that **mimics** the physical world (sensors, actuators, plant). 
![](https://www.ansys.com/content/dam/web/glossary/hardware-in-the-loop-schematic.jpg)
- **ECU**: device that runs the software and provides I/O to the plant it controls[1](https://www.ansys.com/simulation-topics/what-is-hardware-in-the-loop-testing) (like vfd).

Basically: you have a real device, like vfd to control motor used in a elevator, but you lack the real motor (and usually also elevator in that case lol). Instead of connecting vfd to an actual device you don't own, you use **simulated world** that acts like an real life machine (and enviroment factors such as: charge, short-circuts, malfunctions etc) which makes drive 'think' it controls an actual appliance.

`real controlling device + make-believe machine/enviroment`
(in some cases HIL uses a mix of virtual and physical components connected to each other)

> "Simulating a real hardware in the control loop by generating fake signals. This make us test computer response without the need of real sensors and other hardware" [2](https://www.reddit.com/r/explainlikeimfive/comments/k3hxh9/eli5_hardware_in_the_loop_hil/) 

HIL is called "in the loop", because it creates a real-time **closed feedback loop**, mimicking a **continious control cycle** (a system constantly adjusts its output based on measured feedback).

- How it works [3](https://www.mathworks.com/discovery/hardware-in-the-loop-hil.html):
    - Interfacing a real controller hardware with the plant (connections being actual analog and digital I/O), [ a plant has to be an accurate model of physical system replicating the dynamics of the actual system ]
    - Communication between device and virtual system include protocols like TCP, UDP etc 

- Benefits of using HIL:
    - less costly, safer, faster performance, less time consuming
    - in case of any conditions that would cause damage to a real device HIL enables detection of such situations (eg you can simulate firmware sending signal in wrong phase and not detecting it making motor overheat rapidly)

