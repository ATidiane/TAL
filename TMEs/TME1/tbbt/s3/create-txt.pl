#!/usr/bin/perl

use strict;

my @xml_files=glob("xml/*");
foreach my $xml_file(@xml_files){
    #my $xml_file="xml/tbbts03e05.xml";
    
    my @sents=();
    
    open(XML, "<:encoding(UTF-8)", $xml_file);
    
    my $sent="";
    while(<XML>){
	chomp();
	if(/^\s*<w id="[0-9\.]+">(.*)<\/w>/){
	    $sent .= $1." ";
	}
	elsif(/<\/s>/){
	    #$sent .= $_."\n";
	    push(@sents, $sent);
	    $sent="";
	}
	# elsif(/^\s*<s id=/){
	# 	$sent .=  "<s>"." ";
	# }
	
    }
    
    close XML;
    
    my @s = map( { detok($_) } @sents );

    my $txt_file = $xml_file;
    $txt_file  =~ s/xml/txt/g;
     
    open(TXT, ">:encoding(UTF-8)", $txt_file);
    print TXT join("\n", @s);
    close(TXT);
    
}
    
sub detok{
    my ($sent)=@_;
    $sent =~ s/\s+([\.\?,'!])/$1/g;

    return $sent;
}
