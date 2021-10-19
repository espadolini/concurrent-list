//! A lockless append-only unrolled linked list, with a single writer and
//! multiple independent readers.
//!
//! The list is represented by a [`Writer`]-[`Reader`] pair, created by [`new`].
//! The data is stored in contiguous chunks of increasing size, allocated as
//! needed by [`Writer::push()`] and stored in a linked list. Pushing a new
//! value in the list is a `O(1)` operation.
//!
//! The `Reader` can be cloned and can iterate over the data, yielding pinned
//! references to each element in order. The iteration gracefully stops at the
//! end of the list and can be resumed. Advancing the iteration is a `O(1)`
//! operation.
//!
//! Both the `Writer` and the `Reader`s hold a reference count over the list,
//! which is deallocated when the `Writer` and all the `Reader`s are dropped.

#![no_std]
extern crate alloc;

use alloc::{boxed::Box, sync::Arc};
use core::{
    alloc::{Layout, LayoutError},
    cell::UnsafeCell,
    marker::PhantomData,
    mem::{self, MaybeUninit},
    pin::Pin,
    ptr::{self, NonNull},
    sync::atomic::{AtomicUsize, Ordering},
};

/// Nodes of the list, of fixed size.
#[repr(C)]
struct Node<T> {
    /// Pointer to the next `Node`.
    next: UnsafeCell<*const ()>,
    /// Allocation of the current `Node`.
    data: [MaybeUninit<UnsafeCell<T>>],
}

impl<T> Node<T> {
    /// Allocates a new `Node` if `size` is small enough for it to fit in an
    /// allocation. Returns a [`LayoutError`] if an overflow is encountered
    /// while building the [`Layout`]. The resulting `Node` has a
    /// null-initialized `next` while `data` is left uninitialized.
    fn new(size: usize) -> Result<Box<Self>, LayoutError> {
        // this is the layout for Node<T> of a specific size, as shown by the
        // behavior of CoerceUnsized
        let layout = Layout::new::<UnsafeCell<*const ()>>()
            .extend(Layout::array::<MaybeUninit<UnsafeCell<T>>>(size)?)?
            .0
            .pad_to_align();

        assert_ne!(layout.size(), 0);
        // SAFETY: layout is not zero-sized (it's got a pointer in it at the very least)
        let buf = unsafe { alloc::alloc::alloc(layout) };

        let this = Self::from_raw_with_size_mut(buf.cast(), size);

        // SAFETY: we just allocated `this`, it's ok to "dereference"
        let this_next = unsafe { ptr::addr_of!((*this).next) };

        // SAFETY: raw_get is specifically for initializing UnsafeCells
        unsafe { UnsafeCell::raw_get(this_next).write(ptr::null()) };

        // SAFETY: this points to a globally-allocated, initialized Self
        Ok(unsafe { Box::from_raw(this) })
    }

    /// Returns a `*const Node` with size `size` given a `*const ()` to its
    /// allocation.
    fn from_raw_with_size(this: *const (), size: usize) -> *const Self {
        ptr::slice_from_raw_parts(this, size) as *const Self
    }

    /// Returns a `*mut Node` with size `size` given a `*mut ()` to its
    /// allocation.
    fn from_raw_with_size_mut(this: *mut (), size: usize) -> *mut Self {
        ptr::slice_from_raw_parts_mut(this, size) as *mut Self
    }
}

/// Determines the size of the first (and second) node of [`List<T>`]; the size
/// is `1 << SHIFT`.
const SHIFT: usize = 4;

/// Returns the size of the node that contains the `index`-th element of the
/// `List`; the first two chunks have `1 << SHIFT` elements, then the chunk
/// size doubles every time.
fn size_from_index(index: usize) -> usize {
    ((index + 1).next_power_of_two() >> 1).max(1 << SHIFT)
}

/// Returns the relative index in the [`Node::data`] array of the globally
/// `index`-th element of the `List`.
fn index_in_node(index: usize) -> usize {
    if index < (1 << SHIFT) {
        index
    } else {
        index - size_from_index(index)
    }
}

/// The inner object refcounted by `Reader` and `Writer`.
struct List<T> {
    /// Pointer to the first node of the list, if `size > 0`.
    head: UnsafeCell<*const ()>,
    /// Current size of the list.
    size: AtomicUsize,

    /// Marker for typechecking purposes.
    _phantom: PhantomData<*mut Node<T>>,
}

/// Write half of a `concurrent_list`; can only [`push()`] a value to the end of
/// the list, allocating new chunks if necessary.
///
/// [`push()`]: Writer::push
pub struct Writer<T> {
    /// Reference to the [`List`] we can write to.
    inner: Arc<List<T>>,
    /// Pointer to the last node of the list.
    tail: *const (),
}

/// Read half of a `concurrent_list`; it can be cloned and it can be iterated
/// over, to go through the elements of the list.
#[derive(Clone)]
pub struct Reader<T> {
    /// Reference to the [`List`] we can read from.
    inner: Arc<List<T>>,
}

/// Creates a [`Writer`] and a [`Reader`] for the same `concurrent_list`; the
/// `Writer` can be used to append to the list, the `Reader` can be cloned and
/// can iterate over the current contents of the list.
///
/// The list doesn't allocate until a value is pushed in the list.
pub fn new<T>() -> (Writer<T>, Reader<T>) {
    // head must be initialized even if we won't ever read it before it's
    // written onto
    let inner = Arc::new(List {
        head: UnsafeCell::new(ptr::null()),
        size: AtomicUsize::new(0),
        _phantom: PhantomData,
    });

    (
        Writer {
            inner: inner.clone(),
            tail: ptr::null(),
        },
        Reader { inner },
    )
}

impl<T> Writer<T> {
    /// Appends `value` to the list.
    pub fn push(&mut self, value: T) {
        if mem::size_of::<T>() == 0 {
            assert!(self.inner.size.load(Ordering::Relaxed) < usize::MAX);
            // SAFETY: value is zero-sized
            unsafe { NonNull::<T>::dangling().as_ptr().write(value) };
            self.inner.size.fetch_add(1, Ordering::Release);
            return;
        }

        let index = self.inner.size.load(Ordering::Relaxed);
        let node_index = index_in_node(index);

        if index == 0 {
            let new_node: *mut Node<T> = Box::into_raw(Node::new(size_from_index(index)).unwrap());

            // SAFETY: the size is not big enough for any Reader to be looking
            // in this yet
            *unsafe { &mut *self.inner.head.get() } = new_node.cast();

            self.tail = new_node.cast();
        } else if node_index == 0 {
            let old_node = Node::from_raw_with_size(self.tail, size_from_index(index - 1));
            // SAFETY: self.tail points to a valid Node<T> that contains the
            // absolute index `index - 1`
            let old_node: &Node<T> = unsafe { &*old_node };

            let new_node: *mut Node<T> = Box::into_raw(Node::new(size_from_index(index)).unwrap());

            // SAFETY: the size is not big enough for any Reader to be looking
            // in this yet
            *unsafe { &mut *old_node.next.get() } = new_node.cast();

            self.tail = new_node.cast();
        }

        let node = Node::from_raw_with_size(self.tail, size_from_index(index));
        // SAFETY: self.tail points to a valid Node<T> that contains the
        // absolute index `index`
        let node: &Node<T> = unsafe { &*node };

        let raw_cell = node.data[node_index].as_ptr();
        // SAFETY: raw_get is specifically for initializing UnsafeCells
        unsafe { UnsafeCell::raw_get(raw_cell).write(value) };

        self.inner.size.fetch_add(1, Ordering::Release);
    }
}

impl<T> Reader<T> {
    /// Returns an iterator over the list.
    pub fn iter(&self) -> Iter<'_, T> {
        Iter {
            reader: self,
            next: 0,
            node: ptr::null(),
        }
    }

    /// Returns the number of elements in the list.
    pub fn len(&self) -> usize {
        self.inner.size.load(Ordering::Relaxed)
    }

    /// Returns `true` if the list has no elements.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, T> IntoIterator for &'a Reader<T> {
    type IntoIter = Iter<'a, T>;
    type Item = Pin<&'a T>;

    fn into_iter(self) -> Iter<'a, T> {
        self.iter()
    }
}

/// Iterator over the list. The iteration returns None when there's currently no
/// more item in the list, but can be resumed at a later time.
///
/// The iterator can be copied to maintain references to old positions in the
/// list.
#[derive(Clone, Copy)]
pub struct Iter<'a, T> {
    /// The [`Reader`] we're iterating on.
    reader: &'a Reader<T>,
    /// The index that we're about to return.
    next: usize,
    /// Pointer to the current [`Node`].
    node: *const (),
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = Pin<&'a T>;

    fn next(&mut self) -> Option<Pin<&'a T>> {
        // can only increase over time
        let size = self.reader.inner.size.load(Ordering::Acquire);

        assert!(self.next <= size);

        if self.next == size {
            return None;
        }

        if mem::size_of::<T>() == 0 {
            self.next += 1;
            return Some(unsafe { Pin::new_unchecked(NonNull::dangling().as_ref()) });
        }

        let index = self.next;
        let node_index = index_in_node(index);

        if index == 0 {
            // SAFETY: size is greater than 0, nobody is writing to inner.head anymore
            self.node = unsafe { *self.reader.inner.head.get() };
        } else if node_index == 0 {
            let old_node = Node::from_raw_with_size(self.node, size_from_index(index - 1));
            // SAFETY: self.node points to a valid Node<T> that contains the
            // absolute index `index - 1`
            let old_node: &Node<T> = unsafe { &*old_node };

            // SAFETY: we're at the end of old_node and size is big enough for
            // us to be here, so the writer has written old_node.next already
            self.node = unsafe { *old_node.next.get() };
        }

        let node = Node::from_raw_with_size(self.node, size_from_index(index));
        // SAFETY: self.node points to a valid Node<T> that contains the
        // absolute index `index`
        let node: &Node<T> = unsafe { &*node };

        // SAFETY: self.size is incremented by the Writer only *after*
        // initializing the `self.size`-th value, so this must be initialized
        let value = unsafe { node.data[node_index].assume_init_ref() };

        // SAFETY: the Writer is no longer going to touch this and all that
        // Readers do is read
        let value = unsafe { &*value.get() };

        // SAFETY: we will only move the value during the Drop of the List, so
        // we're fulfilling the Pin contract
        let value = unsafe { Pin::new_unchecked(value) };

        self.next += 1;

        Some(value)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.reader.len() - self.next, None)
    }
}

impl<T> Drop for List<T> {
    fn drop(&mut self) {
        let mut size = self.size.swap(0, Ordering::Relaxed);
        let head = mem::replace(self.head.get_mut(), ptr::null());

        if mem::size_of::<T>() == 0 {
            for _ in 0..size {
                unsafe { NonNull::<T>::dangling().as_ptr().drop_in_place() };
            }
            return;
        }

        let mut head = Node::from_raw_with_size_mut(head as *mut (), 1 << SHIFT);

        if !head.is_null() {
            // SAFETY: head points to a valid Node<T>
            let mut node: Box<Node<T>> = unsafe { Box::from_raw(head) };

            let initialized = node.data.len().min(size);
            for v in &mut node.data[0..initialized] {
                // SAFETY: the values are initialized in order in the list, so
                // we have up to size items to drop here; we're dropping in
                // place because we yield Pin<&T>s in the Reader, so we must not
                // move the values
                unsafe { v.as_mut_ptr().drop_in_place() };
            }
            size -= initialized;

            head = Node::from_raw_with_size_mut(*node.next.get_mut() as *mut (), 1 << SHIFT);
        }

        while !head.is_null() {
            // SAFETY: head points to a valid Node<T>
            let mut node: Box<Node<T>> = unsafe { Box::from_raw(head) };

            let initialized = node.data.len().min(size);
            for v in &mut node.data[0..initialized] {
                // SAFETY: the values are initialized in order in the list, so
                // we have up to size items to drop here; we're dropping in
                // place because we yield Pin<&T>s in the Reader, so we must not
                // move the values
                unsafe { v.as_mut_ptr().drop_in_place() };
            }
            size -= initialized;

            head =
                Node::from_raw_with_size_mut(*node.next.get_mut() as *mut (), node.data.len() * 2);
        }
    }
}

unsafe impl<T: Send + Sync> Send for Writer<T> {}
unsafe impl<T: Send + Sync> Sync for Writer<T> {}

unsafe impl<T: Send + Sync> Send for Reader<T> {}
unsafe impl<T: Send + Sync> Sync for Reader<T> {}

unsafe impl<T: Send + Sync> Send for Iter<'_, T> {}
unsafe impl<T: Send + Sync> Sync for Iter<'_, T> {}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn basic() {
        let (mut writer, reader) = new();

        for i in 0_usize..5000 {
            writer.push(i);
        }

        let reader_2 = reader.clone();
        for (i, p) in reader_2.iter().enumerate() {
            assert_eq!(i, *p);
        }

        for (i, p) in reader.iter().enumerate() {
            assert_eq!(i, *p);
        }
    }

    #[test]
    fn drop_count() {
        let (mut writer, reader) = new();

        let sentinel = Arc::new(());

        // halfway through a block
        for _ in 0..((8 << SHIFT) + 6) {
            writer.push(sentinel.clone());
        }

        let sentinel = Arc::try_unwrap(sentinel).unwrap_err();

        drop(reader);
        drop(writer);

        Arc::try_unwrap(sentinel).unwrap();
    }
}
