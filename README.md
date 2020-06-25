# Setup

First, install [Julia](https://julialang.org/downloads/).

Clone the repository to a location of your desire:
```
git clone https://github.com/femtomc/SMILE.jl
```

Enter the directory:
```
cd SMILE.jl
```

Execute at the command line (https):
```
julia -e 'using Pkg; Pkg.activate("."); Pkg.add(url="https://github.com/p2t2/Scruff.git", rev="2d51a04558ff52c20f283e18c6b77d9c43978513"); Pkg.instantiate()'
```

or SSH:
```
julia -e 'using Pkg; Pkg.activate("."); Pkg.add(url="git@github.com:p2t2/Scruff.git", rev="2d51a04558ff52c20f283e18c6b77d9c43978513"); Pkg.instantiate()'
```

This will instantiate your local Julia environment in that directory, as well as pull `Scruff` for use. The `develop` call puts a local copy of `Scruff` in your `HOME/.julia/dev` folder and checks out the `development` branch.

Finally, to run the model on a `.txt` output, use the following command:
```
julia --project=. src/SMILE.jl <your_txt_file>
```

This points the model file at the text file and executes it.

---

Setup all together:
```
git clone https://github.com/femtomc/SMILE.jl
cd SMILE.jl
julia -e 'using Pkg; Pkg.activate("."); Pkg.add(url="https://github.com/p2t2/Scruff.git", rev="2d51a04558ff52c20f283e18c6b77d9c43978513"); Pkg.instantiate()'
```
