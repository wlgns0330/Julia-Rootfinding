using Documenter
using YRoots

makedocs(
    sitename = "YRoots.jl",
    authors  = "BYU Math",
    modules  = [YRoots],
    format   = Documenter.HTML(
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical  = "https://wlgns0330.github.io/Julia-Rootfinding",
        repolink   = "https://github.com/wlgns0330/Julia-Rootfinding",
        # Set explicitly: Documenter shells out to `git remote` to infer this and, when
        # that fails (as it does in a fresh CI checkout), silently defaults to "master".
        # This repository's default branch is main, so every "Edit on GitHub" link would
        # have 404'd.
        edit_link  = "main",
        # Shares the landing page's palette, type and logo, so the docs read as
        # part of the same project. Documenter emits these after its own theme
        # stylesheets, so jroots.css overrides them.
        assets = [
            asset(
                "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600" *
                "&family=IBM+Plex+Sans:wght@400;500;600;700" *
                "&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;" *
                "1,6..72,400;1,6..72,500&display=swap",
                class = :css,
            ),
            "assets/jroots.css",
        ],
        footer = "Part of the [yroots project](https://wlgns0330.github.io/RootFinding/) · " *
                 "[Source](https://github.com/wlgns0330/Julia-Rootfinding) · " *
                 "[Paper (arXiv)](https://arxiv.org/abs/2401.02114)",
    ),
    repo = "https://github.com/wlgns0330/Julia-Rootfinding/blob/{commit}{path}#{line}",
    pages = [
        "Home"        => "index.md",
        "solve"       => "solve.md",
        "Polynomials" => "polynomials.md",
    ],
    # Only the exported API has to appear in an @docs block. The solver internals carry
    # docstrings too, but listing all 156 of them is not what this site is for, and
    # :all would fail the build for every one left out.
    checkdocs = :exports,
)
