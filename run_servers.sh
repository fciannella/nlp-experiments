#!/bin/bash
/etc/init.d/ssh start
jupyter nbextensions_configurator enable --user
jupyter contrib nbextension install --user
jupyter nbextension enable codefolding/main
jupyter nbextension enable collapsible_headings/main
jupyter nbextension enable --py widgetsnbextension
jupyter notebook --allow-root --ip=0.0.0.0