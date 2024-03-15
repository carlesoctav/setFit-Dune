sudo apt update 
sudo apt install -y unzip xvfb libxi6 libgconf-2-4

wget --no-verbose -O /tmp/chrome.deb https://dl.google.com/linux/chrome/deb/pool/main/g/google-chrome-stable/google-chrome-stable_114.0.5735.90-1_amd64.deb
sudo apt install -y /tmp/chrome.deb
sudo rm tmp/chrome.deb

wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip 
unzip chromedriver_linux64.zip

sudo mv chromedriver /usr/bin/chromedriver 
sudo chown root:root /usr/bin/chromedriver 
sudo chmod +x /usr/bin/chromedriver
