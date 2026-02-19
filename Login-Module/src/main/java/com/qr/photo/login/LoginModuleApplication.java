package com.qr.photo.login;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.annotation.Bean;
import org.springframework.http.HttpStatus;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.oauth2.core.user.OAuth2User;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.authentication.HttpStatusEntryPoint;
import org.springframework.security.web.bind.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Collections;
import java.util.Map;

@SpringBootApplication
@RestController
public class LoginModuleApplication {

//	@Bean
//	public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
//		// @formatter:off
//		http
//				.authorizeHttpRequests(a -> a
//						.requestMatchers("/", "/error", "/webjars/**").permitAll()
//						.anyRequest().authenticated()
//				)
//				.exceptionHandling(e -> e
//						.authenticationEntryPoint(new HttpStatusEntryPoint(HttpStatus.UNAUTHORIZED))
//				)
////				.oauth2Login();
//				.oauth2Login(org.springframework.security.config.Customizer.withDefaults());
//		// @formatter:on
//		return http.build();
//	}

	public static void main(String[] args) {
		SpringApplication.run(LoginModuleApplication.class, args);
	}

}
