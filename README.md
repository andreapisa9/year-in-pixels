# 📆 Year in Pixels
## Data driven visualizations that summarize your year!

DISCLAIMER: This repo is free to use and open source, BUT it's licensed with special conditions. Please, read the LICENSE file carefully before using it. If you intentionally break the terms and conditions, I WILL track you down and sue you up the wazoo 😊

DISCLAIMER PT. 2: the code is temporarily **only functional on Macs**. See the bottom of this file to know why and what you can do to help me make it fully portable.

## How to Use

I tried to make my code as easy to use as possible, just know that there is a little bit of work to do, though. Room for improvement!

This repo works alongside the Pixels app ([App Store](https://apps.apple.com/us/app/mood-tracker-by-pixels/id1668460700), [Play Store](https://play.google.com/store/apps/details?id=ar.teovogel.yip)), developed independently by [Teo Vogel](https://teovogel.me/). Pixels is a personal life tracker, it is free and doesn't sell or share your data with third parties. Please make sure to track those aspects of life you value or feel are useful for you! It's very beneficial. 

Once you have enough data tracked (mind that the code is designed to be used once every year), clone this repository on your local machine or download its raw version if you prefer. If you don't know how to do that, just search "git clone tutorial" online: there is plenty of tutorials on how to use Git out there.

### What I track on Pixels

This is how I've personally used Pixels for the past couple years:
1. At the end of every day, I rate how well I felt from 1 to 5 (the app allows you to rate your mood multiple times a day, but I prefer just recording one overall rating);
2. I annotate some tags that describe my day (specifically: "Emotions", "Activities", "Location", "Productivity Rating", "Symptoms" and "Medications"). This is the most personal step of them all, you can customize the tags you use according to what you feel is important to track. Mind you, the code will only work if you use the same tags I use. I'm planning to make it more customizable in the future, but I don't know when I'll be able to come around to it. If you want to contribute and you are skilled enough to do so, feel free to drop a Pull Request and I'll make sure to check it!
3. Finally, I write down a short summary of what happened in the day's "Notes" section. Sometimes, I use this section as a safe space to reflect: it's very therapeutic! Most importantly, I know for sure that my data isn't being shared with anyone for shady purposes, so I can freely let go and share whatever I like.

### Obtaining your Pixels Data

In order to use the code in this repo, you will have to export data from your Pixels app. You can do so by going to the Settings, scrolling down to the bottom and clicking "Export Pixels". You will obtain a JSON file. Once you have it, you should create a folder named `data` in your local repo clone (if you don't know what a repo clone is, you haven't googled "git clone tutorial" yet and should do so) and place the JSON file inside the `data` folder. Rename the file as follows: `data_YEAR.json`. Replace `YEAR` with the year you want to analyze (use 4 digits).

#### At the end of this step, you should have:

- A local repo clone
    - Containing a `data` folder (alongside all downloaded content)
        - The `data` folder should contain a `data_YEAR.json` file with your Pixels export.

### Installing the Conda Environment

For the code to work, you will need to install some version of Anaconda (I suggest Miniconda, which is lighter but has everything you need). Once again, if you don't know how to do that, just search "install miniconda [YOUR OS]" online. Substitute [YOUR OS] with either Windows, your Linux distro of choice or MacOS.

Once you have installed Anaconda, you should open a Terminal console within your local repo clone, and launch the command

```bash
conda env create -f ./environment.yml
```

This command will install all the code libraries needed to make the code work properly. To activate the environment and use it, you will then need to run the command

```bash
conda activate pixels
```
After activation, please check whether the installation of the libraries was correctly completed by comparing the list of libraries you find in the `environment.yml` file to what is installed in the `pixels` environment, by running the command

```bash
conda list
```

#### At the end of this step, you should have:

- A project folder structured as shown in the previous step;
- A `conda` environment called `pixels` with 12 libraries installed.

### Installing an IDE

The final installation you will need to run the code consists of a program which is able to open Jupyter Notebook files. You can choose among a large pool of such programs. I suggest to directly use Jupyter Notebook itself. It's not the easiest to use, but it's on the easy side of the spectrum and won't install any bloatware you don't need. For guides on how to install Jupyter Notebook, you can search "install jupyter notebook [YOUR OS]" online, just like before.

#### At the end of this step, you should have:

- A project folder with the code and your Pixels data export as shown two steps ago;
- A working `conda` environment as shown in the previous step;
- An Interactive Development Environment able to process Jupyter Notebook files, such as Jupyter Notebook itself.

### Generating the Visualizations

Finally, you are able to run the code! To do so, you should open Jupyter Notebook in a Terminal console; from Jupyter, you then need to open the `visualizations.ipynb` file. Once you have done that, you should select the `pixels` environment as working kernel on the top-right of the page, and finally, you can run every code cell of the notebook, by either clicking the "Play" button you find on the left of each cell or by clicking the "Fast-forward" button once (you can find it on the top of the page).

## Support

If you find any problem with the code that you aren't able to overcome on your own or with the help of ChatGPT, come back to this webpage and open an Issue describing your problem (you don't kow how to open an issue? Search "github how to open issue" online). I will try to come back to you as soon as I can to support and fix the code if needed. Mind you this is a passion project, so I won't be able to dedicate much time to it. I'll come around to fixing things whenever possible.

If you are able to fix the code on your own, please open an Issue where you describe both problem and solution. I'll check it out and if I find it to be a meaningful contribution to the project, I'll let you open a Pull Request so you can both publish your fix and appear as a contributor.

## Wanted Contributions

### OS Portability

Right now, the code is **only functional for Macs** due to the emoji font I'm using. I'd like the help of someone who doesn't own just a Mac in order to make the code fully portable to any OS.

# HAVE FUN!