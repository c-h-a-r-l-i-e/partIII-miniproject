import os

num_files = 24

for i in range(0, num_files):
    url = "https://zenodo.org/record/1188976/files/Video_Speech_Actor_{:02d}.zip?download=1".format(i+1)
    filename = "Video_Speech_Actor_{:02d}.zip".format(i+1)

    print("Downloading file {} / {}".format(i+1, num_files))
    os.system("wget {}".format(url))

    # Cut out the last file bits (weird download artifacts
    os.system("mv {}* {}".format(filename, filename))
    os.system("unzip {}".format(filename))
    os.system("rm {}".format(filename))

