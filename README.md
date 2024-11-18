# 📆 Year in Pixels
## Data driven visualizations that summarize your year!

DISCLAIMER: This repo is free to use and open source, BUT it's licensed with special conditions. Please, read the LICENSE file carefully before using it. If you intentionally break the terms and conditions, I WILL track you down and sue you up the wazoo 😊

## How to Use

This repo works alongside the Pixels app ([App Store](https://apps.apple.com/us/app/mood-tracker-by-pixels/id1668460700), [Play Store](https://play.google.com/store/apps/details?id=ar.teovogel.yip)), developed independently by [Teo Vogel](https://teovogel.me/). Pixels is a personal life tracker, it is free and doesn't sell or share your data with third parties. Please make sure to track those aspects of life you value or feel are useful for you! It's very beneficial. 

Once you have enough data tracked (mind that the code is designed to be used once every year), clone this repository on your local machine or download its raw version if you prefer. If you don't know how to do that, just search "git clone tutorial" online: there is plenty of tutorials on how to use Git out there.

### Obtaining your Pixels Data

In order to use the code in this repo, you will have to export data from your Pixels app. You can do so by going to the Settings, scrolling down to the bottom and clicking "Export Pixels". You will obtain a JSON file. Once you have it, you should create a folder named `.env` in your local repo clone (if you don't know what a repo clone is, you haven't googled "git clone tutorial" yet and should do so) and place the JSON file inside the `.env` folder. Rename the file as follows: `data_YEAR.json`. Replace `YEAR` with the year you want to analyze (use 4 digits).

#### At the end of this step, you should have:
- A local repo clone
    - Containing a `.env` folder (alongside all downloaded content)
        - The `.env` folder should contain a `data_YEAR.json` file with your Pixels export.

### Installing the Python Environment
...