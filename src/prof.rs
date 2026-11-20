//! Stage timers for profiling, compiled in by the `profile` feature and
//! empty otherwise.
//!
//! A [`Scope`] adds the time until it is dropped to the entry named by its
//! label and two shape numbers. Scopes do not nest, so the entries add up;
//! run with one worker thread for meaningful figures.

pub use imp::{Scope, report, scope};

#[cfg(feature = "profile")]
mod imp {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::time::{Duration, Instant};

    type Key = (&'static str, usize, usize);

    static TABLE: Mutex<BTreeMap<Key, (Duration, u64, u64)>> = Mutex::new(BTreeMap::new());

    /// A running timer; see the module documentation.
    #[derive(Debug)]
    pub struct Scope {
        key: Key,
        items: usize,
        start: Instant,
    }

    impl Scope {
        /// Stops the timer now rather than at the end of the scope.
        pub fn stop(self) {}
    }

    impl Drop for Scope {
        fn drop(&mut self) {
            let elapsed = self.start.elapsed();
            if let Ok(mut table) = TABLE.lock() {
                let entry = table.entry(self.key).or_default();
                entry.0 += elapsed;
                entry.1 += 1;
                entry.2 += u64::try_from(self.items).unwrap_or(u64::MAX);
            }
        }
    }

    /// Starts timing `name` for a layer of shape `a` by `b` that handles
    /// `items` values (or multiply-adds).
    #[must_use]
    pub fn scope(name: &'static str, a: usize, b: usize, items: usize) -> Scope {
        Scope {
            key: (name, a, b),
            items,
            start: Instant::now(),
        }
    }

    /// The table of everything timed so far, slowest first.
    #[must_use]
    pub fn report() -> String {
        let Ok(table) = TABLE.lock() else {
            return String::new();
        };
        let mut rows: Vec<_> = table.iter().collect();
        rows.sort_by_key(|(_, entry)| std::cmp::Reverse(entry.0));
        let total: Duration = rows.iter().map(|(_, entry)| entry.0).sum();
        let mut lines = vec![format!("profile: {:.3} s in scopes", total.as_secs_f64())];
        for ((name, a, b), (time, calls, items)) in rows {
            let secs = time.as_secs_f64();
            // Picoseconds per item.
            let per_item = (time.as_nanos() * 1000)
                .checked_div(u128::from(*items))
                .unwrap_or(0);
            lines.push(format!(
                "{name:<14} {a:>5} {b:>5} {secs:>9.3} s {:>5.1}% {calls:>9} calls {items:>13} items {:>6}.{:03} ns/item",
                100.0 * secs / total.as_secs_f64().max(1e-12),
                per_item / 1000,
                per_item % 1000,
            ));
        }
        lines.join("\n")
    }
}

#[cfg(not(feature = "profile"))]
mod imp {
    /// A timer that does nothing; see the module documentation.
    #[derive(Debug, Clone, Copy)]
    pub struct Scope;

    impl Scope {
        /// Does nothing without the `profile` feature.
        #[inline]
        pub const fn stop(self) {}
    }

    /// Does nothing without the `profile` feature.
    #[must_use]
    #[inline]
    pub const fn scope(_name: &'static str, _a: usize, _b: usize, _items: usize) -> Scope {
        Scope
    }

    /// Empty without the `profile` feature.
    #[must_use]
    pub const fn report() -> String {
        String::new()
    }
}
