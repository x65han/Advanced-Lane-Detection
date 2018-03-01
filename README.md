# Advanced Lane Detection

<img align="right" src="https://cdn.instructables.com/FNN/AZF7/IG2HFKH0/FNNAZF7IG2HFKH0.LARGE.jpg" width="250px" />
<img src="https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg"/>

- **Industrial Level Lane Detection Algorithm**
- `OpenCV` + `Python`
<br>

---

<div align="center"><b>Lane Detection</b>&emsp;|&emsp;<a href="https://github.com/x65han/Advanced-Lane-Detection/blob/master/outputs/project_video_output.mp4?raw=true">Full Video</a>&emsp;|&emsp;<a href="https://github.com/x65han/Advanced-Lane-Detection/blob/master/report.md">Full Report</a></div><br>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/sample.gif?raw=true"/></div><br>

## Overview

- I used a series of pipeline to better `detect lanes`.
- The pipeline is better explained in this [research paper](http://airccj.org/CSCP/vol5/csit53211.pdf)

<hr>
<div align="center"><b>Undistort Image</b></div>
<img align="center" width="100%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/undistorted.jpg?raw=true" />

<hr>
<div align="center"><b>Unwarp Image | Bird View</b></div>
<img align="center" width="100%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/unwarped.jpg?raw=true" />

<hr>
<div align="center"><b>Apply B-Channel & L-Channel Combined Binary Image</b></div>
<img align="center" width="100%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/pipeline.jpg?raw=true" />

<hr>
<div align="center"><b>Using Historgram to Detect Lane Pixels</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/sliding_window_histogram.jpg?raw=true" /></div>

<hr>
<div align="center"><b>Utilize Sliding Windows to detect Lanes</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/sliding_window.jpg?raw=true" /></div>

<hr>
<div align="center"><b>Highlight Lanes & Smooth Windows</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/fit_lane_highlight.jpg?raw=true" /></div>

<hr>
<div align="center"><b>Reverse Engineer & Apply Lines back</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/draw_lane.jpg?raw=true" /></div>

<hr>
<div align="center"><b>Draw Data</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/draw_data.jpg?raw=true" /></div>

<hr>
<div align="center"><b>Hooray</b></div>
<div align="center"><img width="60%" src="https://github.com/x65han/Advanced-Lane-Detection/blob/master/assets/report/illos-home.jpg?raw=true" /></div>
