use crate::{
    domain::{
        model::CatalogSnapshot,
        run::{CallResult, RunManifest, RunStatus, StoredRun},
    },
    ports::{RunStore, StoreError},
};
use async_trait::async_trait;
use std::{
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

pub struct FilesystemRunStore {
    root: PathBuf,
    write_lock: Mutex<()>,
}

impl FilesystemRunStore {
    pub fn new(root: impl AsRef<Path>) -> Self {
        Self {
            root: root.as_ref().to_path_buf(),
            write_lock: Mutex::new(()),
        }
    }

    fn dir(&self, run_id: &str) -> Result<PathBuf, StoreError> {
        if run_id.is_empty()
            || !run_id
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
        {
            return Err(StoreError::Invalid);
        }
        Ok(self.root.join(run_id))
    }

    fn write_json<T: serde::Serialize>(path: &Path, value: &T) -> Result<(), StoreError> {
        let temp = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec_pretty(value).map_err(|_| StoreError::Invalid)?;
        fs::write(&temp, bytes).map_err(|_| StoreError::Io)?;
        fs::rename(temp, path).map_err(|_| StoreError::Io)
    }
}

#[async_trait]
impl RunStore for FilesystemRunStore {
    async fn create(&self, manifest: &RunManifest) -> Result<(), StoreError> {
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Io)?;
        let dir = self.dir(&manifest.run_id)?;
        if dir.exists() {
            return Err(StoreError::Invalid);
        }
        fs::create_dir_all(&dir).map_err(|_| StoreError::Io)?;
        Self::write_json(&dir.join("manifest.json"), manifest)?;
        File::create(dir.join("results.jsonl")).map_err(|_| StoreError::Io)?;
        Ok(())
    }

    async fn append(&self, run_id: &str, result: &CallResult) -> Result<(), StoreError> {
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Io)?;
        let dir = self.dir(run_id)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("results.jsonl"))
            .map_err(|_| StoreError::Io)?;
        serde_json::to_writer(&mut file, result).map_err(|_| StoreError::Invalid)?;
        file.write_all(b"\n")
            .and_then(|_| file.flush())
            .map_err(|_| StoreError::Io)
    }

    async fn load(&self, run_id: &str) -> Result<StoredRun, StoreError> {
        let dir = self.dir(run_id)?;
        let manifest: RunManifest = serde_json::from_slice(
            &fs::read(dir.join("manifest.json")).map_err(|_| StoreError::NotFound)?,
        )
        .map_err(|_| StoreError::Invalid)?;
        let file = File::open(dir.join("results.jsonl")).map_err(|_| StoreError::Io)?;
        let mut results = Vec::new();
        let mut reader = BufReader::new(file);
        let mut line = Vec::new();
        loop {
            line.clear();
            let count = reader
                .read_until(b'\n', &mut line)
                .map_err(|_| StoreError::Io)?;
            if count == 0 {
                break;
            }
            if line.iter().all(u8::is_ascii_whitespace) {
                continue;
            }
            match serde_json::from_slice::<CallResult>(&line) {
                Ok(result) => results.push(result),
                Err(_) if !line.ends_with(b"\n") => break,
                Err(_) => return Err(StoreError::Invalid),
            }
        }
        Ok(StoredRun { manifest, results })
    }

    async fn set_status(&self, run_id: &str, status: RunStatus) -> Result<(), StoreError> {
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Io)?;
        let dir = self.dir(run_id)?;
        let path = dir.join("manifest.json");
        let mut manifest: RunManifest =
            serde_json::from_slice(&fs::read(&path).map_err(|_| StoreError::NotFound)?)
                .map_err(|_| StoreError::Invalid)?;
        manifest.status = status;
        manifest.finished_at = match status {
            RunStatus::Running | RunStatus::Planned => None,
            _ => Some(chrono::Utc::now()),
        };
        Self::write_json(&path, &manifest)
    }

    async fn save_catalog(
        &self,
        run_id: &str,
        snapshot: &CatalogSnapshot,
    ) -> Result<(), StoreError> {
        let _guard = self.write_lock.lock().map_err(|_| StoreError::Io)?;
        Self::write_json(&self.dir(run_id)?.join("catalog.json"), snapshot)
    }

    async fn catalog(&self, run_id: &str) -> Result<CatalogSnapshot, StoreError> {
        serde_json::from_slice(
            &fs::read(self.dir(run_id)?.join("catalog.json")).map_err(|_| StoreError::NotFound)?,
        )
        .map_err(|_| StoreError::Invalid)
    }
}
