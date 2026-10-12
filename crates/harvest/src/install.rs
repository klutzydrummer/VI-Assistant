//! An inventory of a Legendary Edition install: which games are there and
//! which packages, subtitle files and audio files each has.
//!
//! Layout: `<LE root>/Game/ME1|ME2|ME3/BioGame/CookedPCConsole/*.pcc`, with
//! DLC under `BioGame/DLC/<name>/CookedPCConsole/`. Read-only: nothing here
//! writes to the install.

use std::path::{Path, PathBuf};

use crate::Game;

#[derive(Debug, Clone)]
pub struct FileInfo {
    pub path: PathBuf,
    pub size: u64,
}

#[derive(Debug, Default)]
pub struct GameFiles {
    pub packages: Vec<FileInfo>,
    pub tlks: Vec<FileInfo>,
    pub afcs: Vec<FileInfo>,
}

#[derive(Debug)]
pub struct GameDir {
    pub game: Game,
    /// The game's `BioGame` folder.
    pub biogame: PathBuf,
    pub files: GameFiles,
}

/// Find the games under `root`, which may be the LE root, its `Game`
/// folder, or one game's folder (`.../Game/ME2`).
pub fn scan(root: &Path) -> std::io::Result<Vec<GameDir>> {
    let candidates = [("ME1", Game::Le1), ("ME2", Game::Le2), ("ME3", Game::Le3)];
    let mut games = Vec::new();
    for (folder, game) in candidates {
        for base in [root.join("Game").join(folder), root.join(folder)] {
            let biogame = base.join("BioGame");
            if biogame.is_dir() {
                games.push(GameDir {
                    game,
                    files: collect(&biogame)?,
                    biogame,
                });
                break;
            }
        }
    }
    if games.is_empty() && root.join("BioGame").is_dir() {
        let name = root.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if let Some(&(_, game)) = candidates
            .iter()
            .find(|(f, _)| f.eq_ignore_ascii_case(name))
        {
            let biogame = root.join("BioGame");
            games.push(GameDir {
                game,
                files: collect(&biogame)?,
                biogame,
            });
        }
    }
    Ok(games)
}

fn collect(dir: &Path) -> std::io::Result<GameFiles> {
    let mut files = GameFiles::default();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let kind = entry.file_type()?;
            let path = entry.path();
            if kind.is_dir() {
                stack.push(path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .map(str::to_ascii_lowercase);
            let list = match ext.as_deref() {
                Some("pcc") => &mut files.packages,
                Some("tlk") => &mut files.tlks,
                Some("afc") => &mut files.afcs,
                _ => continue,
            };
            list.push(FileInfo {
                size: entry.metadata()?.len(),
                path,
            });
        }
    }
    for list in [&mut files.packages, &mut files.tlks, &mut files.afcs] {
        list.sort_by(|a, b| a.path.cmp(&b.path));
    }
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_games_and_files() {
        let dir = tempfile::tempdir().unwrap();
        let cooked = dir.path().join("Game/ME2/BioGame/CookedPCConsole");
        let dlc = dir
            .path()
            .join("Game/ME2/BioGame/DLC/DLC_HEN_VT/CookedPCConsole");
        std::fs::create_dir_all(&cooked).unwrap();
        std::fs::create_dir_all(&dlc).unwrap();
        std::fs::create_dir_all(dir.path().join("Game/ME3/BioGame")).unwrap();
        std::fs::write(cooked.join("BioD_Nor.pcc"), [0; 10]).unwrap();
        std::fs::write(cooked.join("BIOGame_INT.tlk"), [0; 3]).unwrap();
        std::fs::write(dlc.join("Audio.afc"), [0; 7]).unwrap();
        std::fs::write(dlc.join("readme.txt"), [0; 1]).unwrap();

        let games = scan(dir.path()).unwrap();
        assert_eq!(games.len(), 2);
        assert_eq!(games[0].game, Game::Le2);
        let f = &games[0].files;
        assert_eq!((f.packages.len(), f.tlks.len(), f.afcs.len()), (1, 1, 1));
        assert_eq!(f.packages[0].size, 10);
        assert_eq!(games[1].game, Game::Le3);

        let one = scan(&dir.path().join("Game/ME2")).unwrap();
        assert_eq!(one.len(), 1);
        assert_eq!(one[0].game, Game::Le2);
    }
}
