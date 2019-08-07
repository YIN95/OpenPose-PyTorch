url = "https://www.dropbox.com/s/96fa5ny6iba02sm/body_pose_model.pth?dl=0"
import urllib.request
u = urllib.request.urlopen(url)
data = u.read()
u.close()
print(data)
with open([body_pose_model.pth], "wb") as f :
    f.write(data)git 