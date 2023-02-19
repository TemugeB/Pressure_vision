# Pressure_vision

[![screenshot](https://user-images.githubusercontent.com/36071915/219939162-70653938-5c17-42f7-bdd4-358629e9ab1f.png)](https://www.youtube.com/watch?v=VKjco6BKUCk)


This is a vision based pressure sensor. But this work is not really uploaded for public use. If you want to deploy this yourself or have some questions, send me a message. I've also set up a live demo that works on a prerecorded video. So technically, you can look inside the code and see how everything works. Unfortunately, the code is a bit of a mess because I whipped this up in about two days. 

**How to**

**Step 1.**

Clone repository

```
git clone https://github.com/TemugeB/Pressure_vision.git
```

**Step 2.**

Get required packages

```
pip3 install -r requirements.txt
```

**Step 3.**

Download video and put in the same folder.
[Video Link](https://drive.google.com/file/d/1XiTIumkK_es4xVtqhl8XpW3hWfUMQWUk/view?usp=share_link)

**Step 4.**

Run the demo. In the mesh window, press "W" to see the mesh edges. Also rotate the mesh to match the video feed.

```
python3 demo_with_video.py samplevid.mp4 
```
