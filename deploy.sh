# ssh-add ~/.ssh/id_rsa
# git remote set-url origin git@github.com:socratesacademy/statsbook.git
# git pull origin main
# git add .
# git commit -m 'this is a message'
# git push origin main
# open atom master branch
jupyter-book build ../statsbook/
# Publish your book's HTML manually to GitHub pages
# publish the _site folder of the main branch to the gh-pages branch
ghp-import -n -p -f _build/html
