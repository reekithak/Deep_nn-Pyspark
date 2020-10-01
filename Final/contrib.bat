@ECHO OFF
git init
git add --all
git commit -m "First try"
git remote add origin https://maindrug.scm.azurewebsites.net:443/maindrug.git
git push -u origin master
PAUSE