# Code for training PNO model of Figure 5

## Requirements

* `docker` and `docker-compose`
* [GIN](https://web.gin.g-node.org/G-Node/Info/wiki/GinCli#quickstart) along with `git` and `git-annex` to download the data. 


## Quickstart

Create folder called `data` in this folder.

```bash
# get the data
cd data
gin login
gin get cajal/Lurz2020 # might take a while; fast internet recommended
cd -

# create docker container (you might need sudo)
docker-compose up jupyter
```
Then open your browser under [localhost:2020](localhost:2020). The password for jupyter lab is `cotton2020`.

Open the notebook and execute the cells. Using a GPU is recommended. 

## Acknowledgements

Some code pieces are reused from (otherwise private) code written by members of [Sinzlab](https://github.com/sinzlab). 
Particular thanks for [Konstantin Willeke](https://github.com/KonstantinWilleke), [Konstantin Lurz](https://github.com/kklurz), and [Edgar Walker](https://github.com/eywalker). 