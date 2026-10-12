//! `vi harvest` on a synthetic install: no game files involved.

use std::path::Path;
use std::process::Command;

use harvest::fixture::{self, PackageBuilder};
use harvest::Game;

fn vi(args: &[&str]) -> (bool, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_vi"))
        .args(args)
        .output()
        .unwrap();
    let text =
        String::from_utf8_lossy(&out.stdout).to_string() + &String::from_utf8_lossy(&out.stderr);
    (out.status.success(), text)
}

fn write(path: &Path, bytes: &[u8]) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, bytes).unwrap();
}

#[test]
fn scans_and_reads_a_synthetic_install() {
    let root = tempfile::tempdir().unwrap();
    let me1 = root.path().join("Game/ME1/BioGame/CookedPCConsole");
    let me2 = root.path().join("Game/ME2/BioGame/CookedPCConsole");

    let mut le1 = PackageBuilder::new(Game::Le1);
    let class = le1.import("Engine", "Class", "BioTlkFile", 0);
    let data = le1.object_data(&[], &fixture::le1_tlk(&[(42, "Keelah se'lai.")]));
    le1.export(class, "GlobalTlk_tlk", 0, 0, data);
    write(&me1.join("Startup_INT.pcc"), &le1.build_compressed(0x40000));

    let mut le2 = PackageBuilder::new(Game::Le2);
    let class = le2.import("Core", "Class", "BioConversation", 0);
    le2.export(class, "tali_dlg", 0, 0, vec![1, 2, 3]);
    write(&me2.join("BioD_Nor_LOC_INT.pcc"), &le2.build_compressed(64));
    write(
        &me2.join("BIOGame_INT.tlk"),
        &fixture::tlk(&[(348112, "Hello.\nShepard.")], &[(348112, "Hi.")]),
    );

    let (ok, out) = vi(&["harvest", "scan", root.path().to_str().unwrap(), "--open"]);
    assert!(ok, "{out}");
    assert!(out.contains("le1 684/171 OodleLeviathan"), "{out}");
    assert!(out.contains("all files read"), "{out}");

    let (ok, out) = vi(&[
        "harvest",
        "package",
        me2.join("BioD_Nor_LOC_INT.pcc").to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    assert!(out.contains("1  BioConversation"), "{out}");

    let tlk = me2.join("BIOGame_INT.tlk");
    let (ok, out) = vi(&["harvest", "tlk", tlk.to_str().unwrap(), "--id", "348112"]);
    assert!(ok, "{out}");
    assert_eq!(out, "348112\tm\tHello.\\nShepard.\n348112\tf\tHi.\n");

    let (ok, out) = vi(&[
        "harvest",
        "tlk",
        me1.join("Startup_INT.pcc").to_str().unwrap(),
    ]);
    assert!(ok, "{out}");
    assert_eq!(out, "42\tGlobalTlk_tlk\tKeelah se'lai.\n");

    // A damaged package is reported, not skipped.
    write(&me2.join("Broken.pcc"), b"garbage");
    let (ok, out) = vi(&["harvest", "scan", root.path().to_str().unwrap()]);
    assert!(!ok);
    assert!(out.contains("Broken.pcc"), "{out}");
}
