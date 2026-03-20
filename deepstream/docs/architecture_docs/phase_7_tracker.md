Now we will be implementing a NVtracker in our deepstream pipeline.

So the idea is the nvtracker runs and shows ID's on the display. But I should have a provision to select a particular ID and then the SUtrack would start tracking that particular object. 

So now what if, the object we were tracking goes out of frame or is occluded for a few frames. Then the nvtracker would assign a new ID to the object. So we need to handle this case. 

So the idea is, we will be using the nvtracker to track the objects. But we will be using the SUtrack to track the object we are interested in and correct this ID if it changes. 

Another case in which the object goes out of frame and then comes back into the frame, it will be assigned a new ID. So we need to handle this case. 

Another case is if the object goes out of the frame and never comes back, then we should stop tracking it and we need to select a new object to track or ingeneral if at any point I want to change the object the SUTrack is tracking, I should be able to do that.

From codeing perspective we should now start working in a another app file, we can take the current deepstream/apps/deepstream_rtsp_app.py as base and start building upon this. The files are getting larger so now we need to shift the necessary utilities in another utils file
 


