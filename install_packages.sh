install_dependencies () {
  apt-get -y install python3-numpy
  if [ $? -ne 0 ]; then
    echo "ERROR: apt-get -y install python3-numpy failed"
    exit 1
  fi
  apt-get -y install python3-scipy
  if [ $? -ne 0 ]; then
    echo "ERROR: apt-get -y install python3-scipy failed"
    exit 1
  fi
  python3 -mpip install matplotlib
  if [ $? -ne 0 ]; then
    echo "ERROR: python3 -mpip install matplotlib failed"
    exit 1
  fi
  pip3 install scikit-learn
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install scikit-learn failed"
    exit 1
  fi
  pip install -U nltk
  if [ $? -ne 0 ]; then
    echo "ERROR: pip install -U nltk failed"
    exit 1
  fi
  pip3 install flask
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install flask failed"
    exit 1
  fi
  pip3 install wtforms
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install wtforms failed"
    exit 1
  fi
  pip3 install seaborn
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install seaborn failed"
    exit 1
  fi
  pip3 install psutil
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install psutil failed"
    exit 1
  fi
  pip3 install progressbar2
  if [ $? -ne 0 ]; then
    echo "ERROR: pip3 install progressbar2 failed"
    exit 1
  fi
}

install_dependencies
