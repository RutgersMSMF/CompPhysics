#!/usr/bin/perl
use Math::Complex;

$Nx = 100;
$Ny = 100;
$max_steps = 1000;

for ($i = 0; $i < $Nx; $i++) {

    for ($j = 0; $j < $Ny; $j++) {
	$x = -2.0 + 3.0 * $i / ($Nx - 1.0);
	$y = -1.0 + 2.0 * $j / ($Ny - 1.0);
	$z0 = $x + $y * i;
	$z = 0;

	for ($itr = 0; $itr < $max_steps; $itr++) {
	
	    if (abs($z) > 2.0) {
			last;
		}

	    $z = $z * $z + $z0;
	
	}

	print "$x  $y  $itr \n";
    
	}
	
}

