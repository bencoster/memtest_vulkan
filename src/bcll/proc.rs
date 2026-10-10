//! Child-process supervision with bounded waits everywhere (Codex BLOCK 89de0c26 #1).
//!
//! * every wait has a deadline; after a timeout the child is killed and its
//!   death is awaited for a bounded grace period: `kill_confirmed` says whether
//!   it actually died (never assumed);
//! * pipe readers run on detached threads and are collected with a bound, so a
//!   grandchild that inherited the pipe cannot hold the supervisor;
//! * containment: on Windows every child joins a kill-on-close Job Object, so if
//!   the supervisor dies its workers die too; on Linux children get
//!   PR_SET_PDEATHSIG=SIGKILL. Workers also enforce their own hard deadline.

use std::io::Read;
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex, mpsc};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    /// how long the child may run
    pub timeout: Duration,
    /// after kill: how long to wait for the death to be observable
    pub kill_grace: Duration,
    /// after exit/kill: how long to wait for pipe readers to reach EOF
    pub pipe_grace: Duration,
    pub poll: Duration,
}

impl Limits {
    pub fn new(timeout: Duration) -> Limits {
        Limits {
            timeout,
            kill_grace: Duration::from_secs(10),
            pipe_grace: Duration::from_secs(5),
            poll: Duration::from_millis(100),
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Outcome {
    /// exit code when the child exited by itself (None: killed / signal / unknown)
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    /// true when the process is known to be gone (exited, or killed and reaped)
    pub dead_confirmed: bool,
    /// error from the kill call, if it failed
    pub kill_error: Option<String>,
    pub stdout: String,
    pub stderr: String,
    /// false when a pipe was still open after `pipe_grace` (descendant holds it)
    pub pipes_complete: bool,
}

impl Outcome {
    /// Human evidence for the output when the death of a timed-out child is unconfirmed.
    pub fn death_note(&self) -> Option<String> {
        (!self.dead_confirmed).then(|| {
            format!(
                "process did NOT confirm death after kill{} - it may still be running on the GPU",
                self.kill_error
                    .as_ref()
                    .map(|e| format!(" (kill error: {e})"))
                    .unwrap_or_default()
            )
        })
    }
}

/// What the supervision loop needs from a child (implemented by std's Child and by test fakes).
pub trait Supervised {
    fn try_wait(&mut self) -> std::io::Result<Option<Option<i32>>>;
    fn kill(&mut self) -> std::io::Result<()>;
}

impl Supervised for Child {
    fn try_wait(&mut self) -> std::io::Result<Option<Option<i32>>> {
        Child::try_wait(self).map(|o| o.map(|s| s.code()))
    }
    fn kill(&mut self) -> std::io::Result<()> {
        Child::kill(self)
    }
}

/// Waits for `child` until `limits.timeout`; then kills and waits `kill_grace`.
/// Returns (exit_code, timed_out, dead_confirmed, kill_error). Never blocks unboundedly.
pub fn supervise(
    child: &mut dyn Supervised,
    limits: &Limits,
) -> (Option<i32>, bool, bool, Option<String>) {
    let start = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(code)) => return (code, false, true, None),
            Ok(None) if start.elapsed() < limits.timeout => std::thread::sleep(limits.poll),
            Ok(None) | Err(_) => break,
        }
    }
    let kill_error = child.kill().err().map(|e| e.to_string());
    let killed_at = Instant::now();
    while killed_at.elapsed() < limits.kill_grace {
        if let Ok(Some(_)) = child.try_wait() {
            return (None, true, true, kill_error);
        }
        std::thread::sleep(limits.poll);
    }
    (None, true, false, kill_error)
}

/// A pipe drained on a detached thread into a shared buffer; `done` fires at EOF.
pub struct Drain {
    buf: Arc<Mutex<Vec<u8>>>,
    done: mpsc::Receiver<()>,
}

impl Drain {
    pub fn start<R: Read + Send + 'static>(mut r: R) -> Drain {
        let buf = Arc::new(Mutex::new(Vec::new()));
        let (tx, done) = mpsc::channel();
        let b = buf.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 8192];
            loop {
                match r.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => b
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend_from_slice(&chunk[..n]),
                }
            }
            let _ = tx.send(());
        });
        Drain { buf, done }
    }

    /// Waits at most `grace` for EOF; returns (text so far, reached_eof).
    pub fn collect(self, grace: Duration) -> (String, bool) {
        let complete = self.done.recv_timeout(grace).is_ok();
        let bytes = self.buf.lock().unwrap_or_else(|e| e.into_inner()).clone();
        (String::from_utf8_lossy(&bytes).into_owned(), complete)
    }
}

/// Runs `cmd` to completion under `limits`, optionally capturing stdout/stderr.
pub fn run(
    mut cmd: Command,
    limits: &Limits,
    capture_stdout: bool,
    capture_stderr: bool,
) -> Result<Outcome, String> {
    cmd.stdin(Stdio::null());
    if capture_stdout {
        cmd.stdout(Stdio::piped());
    }
    if capture_stderr {
        cmd.stderr(Stdio::piped());
    }
    let mut child = spawn_contained(cmd)?;
    let out = child.stdout.take().map(Drain::start);
    let err = child.stderr.take().map(Drain::start);
    let (exit_code, timed_out, dead_confirmed, kill_error) = supervise(&mut child, limits);
    let mut pipes_complete = true;
    let mut grab = |d: Option<Drain>| -> String {
        d.map(|d| {
            let (s, ok) = d.collect(limits.pipe_grace);
            pipes_complete &= ok;
            s
        })
        .unwrap_or_default()
    };
    let stdout = grab(out);
    let stderr = grab(err);
    Ok(Outcome {
        exit_code,
        timed_out,
        dead_confirmed,
        kill_error,
        stdout,
        stderr,
        pipes_complete,
    })
}

/// A long-running child (e.g. telemetry) stopped explicitly with a bounded kill.
pub struct Background {
    child: Child,
}

impl Background {
    pub fn spawn(mut cmd: Command) -> Result<Background, String> {
        cmd.stdin(Stdio::null());
        spawn_contained(cmd).map(|child| Background { child })
    }

    /// Kills the child (if still running) and waits at most `grace` for death.
    pub fn stop(mut self, grace: Duration) -> Outcome {
        let limits = Limits {
            timeout: Duration::ZERO,
            kill_grace: grace,
            pipe_grace: Duration::ZERO,
            poll: Duration::from_millis(50),
        };
        let (exit_code, _, dead_confirmed, kill_error) = supervise(&mut self.child, &limits);
        Outcome {
            exit_code,
            timed_out: false,
            dead_confirmed,
            kill_error,
            pipes_complete: true,
            ..Default::default()
        }
    }
}

/// Worker-side hard limit: after `seconds` the worker records the fact in its events file
/// (if any) and exits with `code`, even if its main thread is stuck in a driver call or the
/// supervisor is gone (Codex BLOCK 89de0c26 #1).
pub fn arm_self_deadline(seconds: Option<f64>, code: i32, events: &std::path::Path) {
    let Some(limit) = seconds.filter(|s| s.is_finite() && *s > 0.0) else {
        return;
    };
    let events = events.to_path_buf();
    let _ = std::thread::Builder::new()
        .name("bcll-self-deadline".into())
        .spawn(move || {
            std::thread::sleep(Duration::from_secs_f64(limit));
            if !events.as_os_str().is_empty() {
                let mut ev = super::events::EventWriter::create(&events);
                ev.emit(
                    "worker_error",
                    serde_json::json!({"message": format!("worker hard deadline {limit}s reached; self-terminated"), "timeout": true}),
                );
            }
            std::process::exit(code);
        });
}

/// Spawns with parent-death containment (see module docs).
pub fn spawn_contained(mut cmd: Command) -> Result<Child, String> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: prctl is async-signal-safe; nothing else runs between fork and exec here.
        unsafe {
            cmd.pre_exec(|| {
                nix::libc::prctl(nix::libc::PR_SET_PDEATHSIG, nix::libc::SIGKILL);
                Ok(())
            });
        }
    }
    let child = cmd.spawn().map_err(|e| format!("spawn failed: {e}"))?;
    #[cfg(windows)]
    windows_job::assign(&child);
    Ok(child)
}

#[cfg(windows)]
mod windows_job {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };

    struct Job(isize);
    // SAFETY: a job handle is a process-wide kernel handle, usable from any thread.
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    /// One kill-on-close job per supervisor process; the handle is never closed explicitly,
    /// so the OS closes it at process exit and kills every still-running member.
    fn job() -> Option<&'static Job> {
        static JOB: OnceLock<Option<Job>> = OnceLock::new();
        JOB.get_or_init(|| unsafe {
            let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if h.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            let ok = SetInformationJobObject(
                h,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const core::ffi::c_void,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            (ok != 0).then_some(Job(h as isize))
        })
        .as_ref()
    }

    pub fn assign(child: &std::process::Child) {
        if let Some(j) = job() {
            // SAFETY: both handles are valid for the duration of the call. Failure (e.g. the
            // supervisor itself runs in a job that forbids nesting) only loses containment;
            // the worker's own hard deadline still applies.
            unsafe {
                AssignProcessToJobObject(j.0 as _, child.as_raw_handle() as _);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A child that never exits by itself; `kill_works` / `dies_after` model the cases.
    struct Fake {
        kill_works: bool,
        dies_after_polls: Option<u32>,
        killed: bool,
        polls_after_kill: u32,
    }

    impl Supervised for Fake {
        fn try_wait(&mut self) -> std::io::Result<Option<Option<i32>>> {
            if self.killed {
                self.polls_after_kill += 1;
                if self
                    .dies_after_polls
                    .is_some_and(|n| self.polls_after_kill >= n)
                {
                    return Ok(Some(None));
                }
            }
            Ok(None)
        }
        fn kill(&mut self) -> std::io::Result<()> {
            self.killed = self.kill_works;
            if self.kill_works {
                Ok(())
            } else {
                Err(std::io::Error::other("access denied"))
            }
        }
    }

    fn quick(timeout_ms: u64, grace_ms: u64) -> Limits {
        Limits {
            timeout: Duration::from_millis(timeout_ms),
            kill_grace: Duration::from_millis(grace_ms),
            pipe_grace: Duration::from_millis(200),
            poll: Duration::from_millis(5),
        }
    }

    #[test]
    fn kill_failure_is_reported_as_unconfirmed_and_bounded() {
        let mut f = Fake {
            kill_works: false,
            dies_after_polls: None,
            killed: false,
            polls_after_kill: 0,
        };
        let t = Instant::now();
        let (code, timed_out, dead, err) = supervise(&mut f, &quick(30, 60));
        assert!(t.elapsed() < Duration::from_secs(2));
        assert_eq!((code, timed_out, dead), (None, true, false));
        assert!(err.unwrap().contains("access denied"));
    }

    #[test]
    fn delayed_death_within_grace_is_confirmed() {
        let mut f = Fake {
            kill_works: true,
            dies_after_polls: Some(3),
            killed: false,
            polls_after_kill: 0,
        };
        let (_, timed_out, dead, err) = supervise(&mut f, &quick(20, 500));
        assert!(timed_out && dead && err.is_none());
    }

    #[test]
    fn death_slower_than_grace_is_unconfirmed() {
        let mut f = Fake {
            kill_works: true,
            dies_after_polls: Some(10_000),
            killed: false,
            polls_after_kill: 0,
        };
        let (_, _, dead, _) = supervise(&mut f, &quick(10, 50));
        assert!(!dead);
        let o = Outcome {
            dead_confirmed: false,
            kill_error: None,
            ..Default::default()
        };
        assert!(o.death_note().unwrap().contains("did NOT confirm death"));
    }

    /// A pipe whose writer never closes (a descendant holding the inherited handle).
    struct Stalled;
    impl Read for Stalled {
        fn read(&mut self, _: &mut [u8]) -> std::io::Result<usize> {
            std::thread::sleep(Duration::from_secs(3600));
            Ok(0)
        }
    }

    #[test]
    fn stalled_pipe_is_abandoned_after_grace() {
        let t = Instant::now();
        let (text, complete) = Drain::start(Stalled).collect(Duration::from_millis(100));
        assert!(!complete && text.is_empty());
        assert!(t.elapsed() < Duration::from_secs(2));
        let (text, complete) = Drain::start(&b"partial output"[..]).collect(Duration::from_secs(2));
        assert!(complete);
        assert_eq!(text, "partial output");
    }

    fn sleeper(secs: u32) -> Command {
        if cfg!(windows) {
            let mut c = Command::new("cmd");
            c.args(["/C", &format!("ping -n {} 127.0.0.1 >NUL", secs + 1)]);
            c
        } else {
            let mut c = Command::new("sh");
            c.args(["-c", &format!("sleep {secs}")]);
            c
        }
    }

    #[test]
    fn real_child_timeout_is_killed_and_confirmed() {
        let t = Instant::now();
        let o = run(sleeper(30), &quick(300, 5000), false, false).unwrap();
        assert!(o.timed_out && o.dead_confirmed, "{o:?}");
        assert!(t.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn real_child_exit_code_and_output() {
        let mut c = if cfg!(windows) {
            Command::new("cmd")
        } else {
            Command::new("sh")
        };
        if cfg!(windows) {
            c.args(["/C", "echo hello & exit 3"]);
        } else {
            c.args(["-c", "echo hello; exit 3"]);
        }
        let o = run(c, &quick(10_000, 1000), true, true).unwrap();
        assert_eq!(o.exit_code, Some(3));
        assert!(o.stdout.contains("hello") && o.pipes_complete && !o.timed_out);
    }

    #[test]
    fn inherited_pipe_held_by_grandchild_does_not_block() {
        // the shell exits at once but a background grandchild keeps stdout open ~20 s
        let mut c = if cfg!(windows) {
            Command::new("cmd")
        } else {
            Command::new("sh")
        };
        if cfg!(windows) {
            c.args(["/C", "start /B ping -n 20 127.0.0.1 & echo started"]);
        } else {
            c.args(["-c", "sleep 20 & echo started"]);
        }
        let t = Instant::now();
        let o = run(c, &quick(10_000, 1000), true, false).unwrap();
        assert!(t.elapsed() < Duration::from_secs(8), "{:?}", t.elapsed());
        assert!(o.dead_confirmed && !o.pipes_complete, "{o:?}");
        assert!(o.stdout.contains("started"));
    }

    #[test]
    fn background_stop_is_bounded() {
        let b = Background::spawn(sleeper(30)).unwrap();
        let t = Instant::now();
        let o = b.stop(Duration::from_secs(5));
        assert!(o.dead_confirmed);
        assert!(t.elapsed() < Duration::from_secs(6));
    }
}
