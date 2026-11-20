/*
 * Fake CPU count for the Vast-conditions rehearsal (jobs/rehearsal/vast-limits.sh).
 *
 * A rented container on a big host sees the host's logical CPUs (48 in r7) even when its CPU
 * quota is far smaller, and OpenBLAS sizes one thread pool per process from that view. This
 * LD_PRELOAD library makes a small home box report FAKE_NPROC CPUs through the calls OpenBLAS and
 * Python use (sched_getaffinity, sysconf, get_nprocs, get_nprocs_conf), so the same thread storm
 * happens at home. It changes nothing else, and it is a no-op when FAKE_NPROC is unset.
 * It is built into ~/.cache/vi-rehearsal, never into the repo and never sent to a worker.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <sched.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int fake(void) {
  const char *s = getenv("FAKE_NPROC");
  int n = s ? atoi(s) : 0;
  return n > 0 ? n : 0;
}

int sched_getaffinity(pid_t pid, size_t sz, cpu_set_t *m) {
  static int (*real)(pid_t, size_t, cpu_set_t *);
  if (!real) real = dlsym(RTLD_NEXT, "sched_getaffinity");
  int n = fake(), r = real(pid, sz, m);
  if (r != 0 || !n) return r;
  memset(m, 0, sz);
  for (int i = 0; i < n && (size_t)i < sz * 8; i++) CPU_SET_S(i, sz, m);
  return 0;
}

long sysconf(int name) {
  static long (*real)(int);
  if (!real) real = dlsym(RTLD_NEXT, "sysconf");
  int n = fake();
  if (n && (name == _SC_NPROCESSORS_ONLN || name == _SC_NPROCESSORS_CONF)) return n;
  return real(name);
}

int get_nprocs(void) {
  int n = fake();
  if (n) return n;
  int (*r)(void) = dlsym(RTLD_NEXT, "get_nprocs");
  return r();
}

int get_nprocs_conf(void) {
  int n = fake();
  if (n) return n;
  int (*r)(void) = dlsym(RTLD_NEXT, "get_nprocs_conf");
  return r();
}
