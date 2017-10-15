LOG
----

Currently a coarse RHF program is finished. But the most important thing is that
cooperating with LIBINT correctly.

###### Current problems:

- The integration of LIBINT (the libint2py API) is not elegant enough. I am forced to use a second sorting algorithm after the generation of the integrals.

- The python API `molecule` is also very simple.

- The RHF program has a inefficient procedure to deal with the eri. Perhaps a better algorithm exists.

###### Next step:

- Finish the first stable version of libint2py. Separathe the HF program out.
